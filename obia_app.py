import sys
import os
import time
import numpy as np
import pandas as pd
import json
import rasterio
import geopandas as gpd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QFileDialog, QTabWidget, QSlider, QListWidget, QListWidgetItem,
    QColorDialog, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QTextEdit,
    QDialog, QInputDialog, QMessageBox, QCheckBox, QRadioButton, QButtonGroup, QScrollArea,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QMenu, QGridLayout,
    QSizePolicy
)
from PySide6.QtGui import QImage, QColor, QPainter, QTextCursor, QIcon, QPixmap, QShortcut, QKeySequence, QAction, QDesktopServices
from PySide6.QtCore import Qt, QThread, Signal, QObject, QUrl, QTimer

from skimage.color import rgb2hsv, hsv2rgb
from skimage.segmentation import find_boundaries
from skimage import exposure
from sklearn.ensemble import RandomForestClassifier

# Import modular components
from obia_engine import ObiaEngine, Cancelled, cuda_available, working_bytes, free_ram_bytes
from obia_ui import InteractivePhotoViewer, ClassListDelegate, ToggleSwitch, CollapsibleSection, render_confusion_matrix_image, render_separability_image
from obia_ui import GroupCard, ChipPickerDialog, make_chip_qimage, chip_strip_pixmap
from obia_ui import ChipGallery, ClassNameCombo, suggest_class_color, rgba_premultiplied, ElidedInfoLabel
import obia_discover as od
from obia_odm_window import OdmWindow, saved_job
import shutil

import app_info

MANUAL_URL = "https://leobarua.github.io/liteclassify/"
import gc
import math

def get_resource_path(relative_path):
    """ Get absolute path to a bundled resource (e.g. the app icon) -- works both for a
    normal `python obia_app.py` dev run and for the PyInstaller-frozen build_exe.py
    output. PyInstaller sets sys._MEIPASS to the bundle's extracted/collected data
    directory at runtime (both --onefile and --onedir); build_exe.py's
    --add-data=<icon>;. places the icon at the root of that directory, matching the
    '.' destination used there. In dev mode, resolve relative to this script's own
    directory rather than the current working directory -- a shortcut or a `python
    some/other/dir/obia_app.py` launch can have a cwd that isn't the project folder,
    which would silently break icon loading (get_resource_path was defined but never
    actually called anywhere before this -- this was dead code with an unfixed cwd bug
    in its dev-mode fallback until now). """
    try:
        # PyInstaller creates a temporary/collected folder and stores its path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# --- Custom Stream Class to Redirect Console Output Safely ---
class SafeStream(QObject):
    """
    Thread-safe stream redirector that marshals console print messages
    across thread boundaries using Qt Signals (Queued Connections).
    Handles windowed environment where stdout/stderr might be None.
    """
    text_written = Signal(str)

    def __init__(self, original_stream):
        super().__init__()
        self.original_stream = original_stream
        self._is_writing = False

    def write(self, text):
        # Guard: In windowed executable mode, the original terminal stream is None
        if self.original_stream is not None:
            try:
                self.original_stream.write(text)
            except Exception:
                pass
                
        if not self._is_writing:
            try:
                self._is_writing = True
                # Safely emit signal to be caught on the main thread
                self.text_written.emit(text)
            except Exception:
                pass
            finally:
                self._is_writing = False

    def flush(self):
        if self.original_stream is not None:
            try:
                self.original_stream.flush()
            except Exception:
                pass


DISPLAY_MAX_PX = 40e6   # overview layers (segment colours, edges, classes, groups) are capped at this
DISPLAY_BASE_MAX_PX = 64e6   # the base ortho image too; zoomed in, a sharp patch is drawn over it


def display_stride(shape):
    """ Row/column step that keeps an overview layer under DISPLAY_MAX_PX pixels. Layers
    are drawn stretched over the full-resolution ortho, so a step of 2-4 on a very large
    image is invisible at normal zoom and saves gigabytes. """
    return max(1, int(math.ceil(math.sqrt(shape[0] * shape[1] / DISPLAY_MAX_PX))))


def bgrx_qimage(rgb_hw3):
    """ (h, w, 3) uint8 RGB -> owned QImage in RGB32, the format QPixmap can share without
    a copy (an RGB888 QImage is converted to a second, 4-byte-per-pixel buffer). """
    h, w = rgb_hw3.shape[:2]
    buf = np.empty((h, w, 4), dtype=np.uint8)
    buf[..., 0] = rgb_hw3[..., 2]
    buf[..., 1] = rgb_hw3[..., 1]
    buf[..., 2] = rgb_hw3[..., 0]
    buf[..., 3] = 255
    return QImage(buf.data, w, h, 4 * w, QImage.Format_RGB32).copy()


def build_boundary_and_visual_caches(engine):
    """ The mean-colour "Segmented" view and the segment-edge overlay, both at display
    resolution (see display_stride) -- they are drawn stretched over the ortho. """
    seg = engine.segmented_image
    s = display_stride(seg.shape)
    seg_d = np.ascontiguousarray(seg[::s, ::s])
    flat = seg_d.ravel()
    n = int(seg.max()) + 1
    counts = np.bincount(flat, minlength=n).astype(np.float64)
    counts[counts == 0] = 1
    lut = np.zeros((n, 3), dtype=np.uint8)
    for b in range(3):
        band = engine.original_ortho_data[b, ::s, ::s].ravel()
        lut[:, b] = np.clip(np.bincount(flat, weights=band, minlength=n) / counts, 0, 255).astype(np.uint8)
    base_seg_qimg = bgrx_qimage(lut[seg_d])

    boundaries = find_boundaries(seg_d, mode='inner')
    h, w = seg_d.shape
    edge = np.zeros((h, w, 4), dtype=np.uint8)
    edge[boundaries] = rgba_premultiplied(QColor(0, 0, 255), 127)
    boundary_overlay_qimg = QImage(edge.data, w, h, 4 * w, QImage.Format_ARGB32_Premultiplied).copy()
    return base_seg_qimg, boundary_overlay_qimg


# --- Background worker threads ---
class SegmentationWorker(QObject):
    # Emits: (segmented_image, base_seg_qimg, boundary_overlay_qimg, num_polygons, duration)
    finished = Signal(object, object, object, int, float)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, engine, n_segments, compactness, sigma, app_instance, force_device="auto"):
        super().__init__()
        self.engine = engine
        self.snapshot = engine.snapshot()     # put back if the run is cancelled
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.app = app_instance
        self.force_device = force_device

    def run(self):
        try:
            start_time = time.time()

            # [Step 1/4] Core Segmentation
            print("\n[Step 1/4] Running spatial-spectral SLIC segmentation...")
            self.engine.segment_image(
                self.n_segments,
                self.compactness,
                self.sigma,
                force_device=self.force_device
            )
            segmented = self.engine.segmented_image
            self.engine.check_cancel()
            self.engine.build_adjacency_graph()

            if segmented is None:
                self.failed.emit("Segmentation engine failed to output a label matrix.")
                return

            # [Step 2/4] Creating boundary display matrices
            print("[Step 2/4] Compiling boundary display caches...")
            base_seg_qimg, boundary_overlay_qimg = build_boundary_and_visual_caches(self.engine)

            # [Step 3/4] Compiling spatial geometries (Polygon tracing)
            self.engine.check_cancel()
            print("[Step 3/4] Vectorizing segments into crisp polygon layers...")
            self.engine.vectorize_segments()
            num_polygons = len(self.engine.segments_gdf)

            # [Step 4/4] Multi-core texture feature extraction
            self.engine.check_cancel()
            print("[Step 4/4] Pre-computing 19D / 23D texture and shape features in parallel...")
            self.engine.compute_all_features_parallel()
            self.engine.check_cancel()
            self.engine.compute_neighbor_relative_features()
            self.engine.compute_height_features()

            duration = time.time() - start_time
            self.finished.emit(segmented, base_seg_qimg, boundary_overlay_qimg, num_polygons, duration)
        except Cancelled:
            self.engine.restore(self.snapshot)
            self.cancelled.emit()
        except Exception:
            import traceback
            self.failed.emit(traceback.format_exc())


class DeepFeatureWorker(QObject):
    """ Computes DINOv2 deep features for every segment off the UI thread (Discover tab). """
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int)
    cancelled = Signal()

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.cancel_requested = False

    def run(self):
        try:
            e = self.engine
            deep = od.compute_deep_features(e.original_ortho_data, e.segmented_image, e.valid_mask,
                                            e.effective_pixel_size or 0.1,
                                            progress=lambda d, n: self.progress.emit(d, n),
                                            cancelled=lambda: self.cancel_requested)
            if deep is None:
                self.cancelled.emit()
            else:
                self.finished.emit(deep)
        except Exception:
            import traceback
            self.failed.emit(traceback.format_exc())


class EditClassDialog(QDialog):
    def __init__(self, current_name, current_color, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Class Settings")
        self.setMinimumWidth(320)
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Class Name:"))
        self.name_edit = QLineEdit(current_name)
        layout.addWidget(self.name_edit)
        
        color_layout = QHBoxLayout()
        self.color_preview = QPushButton()
        self.color_preview.setFixedSize(40, 25)
        self.color_preview.setStyleSheet(f"background-color: {current_color.name()}; border: 1px solid black;")
        
        self.color_btn = QPushButton("Change Color")
        self.color_btn.clicked.connect(self.choose_new_color)
        
        color_layout.addWidget(self.color_preview)
        color_layout.addWidget(self.color_btn)
        layout.addLayout(color_layout)
        
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        self.chosen_color = current_color
        self.chosen_name = current_name
        
    def choose_new_color(self):
        color = QColorDialog.getColor(self.chosen_color, self, "Select Class Color")
        if color.isValid():
            self.chosen_color = color
            self.color_preview.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black;")
            
    def get_values(self):
        return self.name_edit.text().strip(), self.chosen_color


# --- Main window ---
class ObiaApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(app_info.WINDOW_TITLE)
        self.setGeometry(100, 100, 1650, 950)

        # Window icon (top-left of the title bar, and the taskbar icon when running
        # unfrozen via `python obia_app.py` -- the frozen .exe already gets its
        # taskbar/shortcut icon baked in by PyInstaller's --icon flag regardless of
        # this call, see build_exe.py, but Windows still reads the *window's own*
        # icon for the title bar and Alt-Tab thumbnail from whatever Qt sets here).
        icon_path = get_resource_path(app_info.ICON_FILE)
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            print(f"Notice: window icon not found at '{icon_path}' -- using Qt's default.")

        self.engine = ObiaEngine()
        self.workspace_dir = None
        self.classes = {}
        self.classifier = None
        
        self.base_raw_qimg = None
        self.raw_pixmap = None
        self.raw_grid = None          # (w, h) of the working grid the base pixmap stands for
        self.classified_qimg = None
        self._seg_disp_cache = None
        self._seg_slices_cache = None
        self._samples_item = None
        self._sel_key = None
        self.base_seg_qimg = None
        self.raw_display_rgb = None
        self.raw_pixmap = None
        self.classified_qimg = None
        self.classified_image_data = None
        self.raw_classified_raster_ints = None
        self.sieved_classified_raster_ints = None
        self.boundary_overlay_qimg = None
        self.hillshade_qimg = None  # rebuilt from engine.dsm_data whenever DSM loads/unloads
        self._syncing_resample_controls = False  # reentrancy guard, see _on_resample_scale_changed
        self.current_view_mode = "segmented"
        self.sample_class_map = None  # Fast incremental lookup map to prevent np.isin bottlenecks

        self.selection_mode = False
        self.selected_segments = set()
        self.locked_selection = set()      # segments locked in by a previous click/paint, fixed IDs
        self.current_seed_segments = set()  # the newest click's seed(s), grown live by the tolerance slider
        self.selection_undo_stack = []
        self._selection_drag_active = False

        # --- Accuracy Assessment (Tab 3) state ---
        self.acc_checked_point_ids = set()  # checkbox-selected point ids, batch-assign target
        self.acc_nav_index = -1             # index into the currently-filtered list, for </> navigation
        self.acc_point_move_id = None       # point_id currently being Ctrl+dragged, or None
        self.iterations = []                # snapshots taken at each Run Accuracy Assessment, see _record_iteration

        # --- Discover Classes (Tab 2) state, see _discover_reset ---
        self.deep = None
        self.groups = None
        self.disc_selected = None
        self.disc_multi = []
        self.disc_chip_offset = 0
        self._seg_index = None
        self._sim_space = None
        self._strip_cache = {}
        self._disc_cards = {}

        self.init_main_layout()

        self.stdout_redirector = SafeStream(sys.__stdout__)
        self.stdout_redirector.text_written.connect(self.stdout_received)
        sys.stdout = self.stdout_redirector

        self.stderr_redirector = SafeStream(sys.__stderr__)
        self.stderr_redirector.text_written.connect(self.stdout_received)
        sys.stderr = self.stderr_redirector

        print(f"{app_info.APP_NAME} v{app_info.APP_VERSION} by {app_info.APP_AUTHOR}. Load an image to begin.")

    def stdout_received(self, text):
        self.console.moveCursor(QTextCursor.End)
        self.console.insertPlainText(text)

    # Tabs 2-4: the same two left-column widths, so the map keeps its place and size when
    # switching tabs (Tab 1 has one column). Column 2 is wide enough for Discover's tiles.
    LEFT_COL1_W = 360
    LEFT_COL2_W = 440

    def init_main_layout(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        self._map_infos = []        # image info + cursor coordinates, one per tab (see _new_map_info)
        self._map_info_key = None
        self._map_info_line = ""
        self._cursor_text = ""
        self._crs_name_cache = {}
        self._shared_view = None    # zoom/pan shared by the four maps (see _on_view_changed)
        self._map_headers = []      # control strips above / hint lines below each map, equal height
        self._map_tips = []

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(140)
        self.console.setStyleSheet("background-color: #1e1e1e; color: #a9dc76; font-family: Consolas, monospace;")
        main_layout.addWidget(self.console)

        self.init_load_view_tab()
        self.init_discover_tab()
        self.init_classification_tab()
        self.init_accuracy_tab()
        self.init_menu_bar()
        self._saved_sig = self._state_signature()

        # WebODM: a job left running by an earlier session shows in the status bar and is
        # picked up again (the window itself stays closed until asked for).
        self.odm_window = None
        self.odm_status_btn = QPushButton()
        self.odm_status_btn.setFlat(True)
        self.odm_status_btn.setToolTip("Show the WebODM window")
        self.odm_status_btn.clicked.connect(self.open_odm_window)
        self.odm_status_btn.setVisible(False)
        self.statusBar().addPermanentWidget(self.odm_status_btn)
        if saved_job():
            self._ensure_odm_window()

        # Every map gets the tallest header (Discover's two rows) and the same hint line,
        # so the map area keeps its size from tab to tab.
        for group in (self._map_headers, self._map_tips):
            h = max(w.sizeHint().height() for w in group)
            for w in group:
                w.setFixedHeight(h)

        for v in self._map_viewers():
            v.viewChanged.connect(lambda v=v: self._on_view_changed(v))
            v.cursorMoved.connect(self._on_map_cursor)
            v.cursorLeft.connect(self._on_map_cursor_left)
        self.tabs.currentChanged.connect(lambda _: QTimer.singleShot(0, self._apply_shared_view))
        self._map_info_timer = QTimer(self)
        self._map_info_timer.setInterval(1000)
        self._map_info_timer.timeout.connect(self._update_map_info)
        self._map_info_timer.start()

    def init_menu_bar(self):
        """ File / Import / About menus -- workspace actions live here instead of a button
        on every tab, as in a regular Windows application. """
        mb = self.menuBar()
        file_menu = mb.addMenu("&File")
        self.act_new = QAction("&New", self)
        self.act_new.setShortcut(QKeySequence("Ctrl+N"))
        self.act_new.setStatusTip("Start a blank workspace (asks first if there are unsaved changes)")
        self.act_new.triggered.connect(self.new_workspace)
        self.act_load_workspace = QAction("&Load Workspace…", self)
        self.act_load_workspace.setShortcut(QKeySequence("Ctrl+O"))
        self.act_load_workspace.triggered.connect(self.open_workspace_dialog)
        self.act_save_workspace = QAction("&Save Workspace", self)
        self.act_save_workspace.setShortcut(QKeySequence("Ctrl+S"))
        self.act_save_workspace.triggered.connect(self.save_workspace_session)
        self.act_save_workspace_as = QAction("Save Workspace &As…", self)
        self.act_save_workspace_as.setShortcut(QKeySequence("Ctrl+Alt+S"))
        self.act_save_workspace_as.triggered.connect(self.save_workspace_as)
        for a in (self.act_new, self.act_load_workspace, self.act_save_workspace, self.act_save_workspace_as):
            file_menu.addAction(a)
        file_menu.insertSeparator(self.act_save_workspace)

        import_menu = mb.addMenu("&Import")
        self.act_import_webodm = import_menu.addAction("Orthomosaic from Drone &Photos (WebODM)…",
                                                       self.open_odm_window)
        self.act_import_webodm.setStatusTip("Process a folder of drone photos with your WebODM installation")
        import_menu.addSeparator()
        self.act_import_ortho = import_menu.addAction("&Orthomosaic…", self.browse_for_ortho)
        self.act_import_dsm = import_menu.addAction("&DSM…", self.browse_for_dsm)
        import_menu.addSeparator()
        self.act_import_csv = import_menu.addAction("&Ground Truth CSV…", self.import_ground_truth_csv)

        about_menu = mb.addMenu("&About")
        self.act_manual = about_menu.addAction("&Manual", lambda: QDesktopServices.openUrl(QUrl(MANUAL_URL)))
        self.act_manual.setStatusTip(MANUAL_URL)

    # --- unsaved-change tracking ---
    def _state_signature(self):
        """ A fingerprint of everything Save Workspace writes, to tell whether there is
        unsaved work: inputs, segmentation, classes and their samples, groups, accuracy
        points, and the classified map. """
        e = self.engine
        if getattr(e, "ortho_path", None) is None:
            return None
        parts = [e.ortho_path, e.dsm_path, e.effective_pixel_size,
                 None if e.segmented_image is None else od.seg_fingerprint(e.segmented_image),
                 tuple((n, d["color"].name(), hash(frozenset(int(x) for x in d["training_segments"])))
                       for n, d in self.classes.items()),
                 None if self.groups is None else json.dumps(self.groups.to_json(), sort_keys=True),
                 id(self.raw_classified_raster_ints)]
        gdf = e.accuracy_points_gdf
        if gdf is not None and len(gdf):
            parts.append(tuple(zip(gdf["point_id"].tolist(), gdf["reference_class"].astype(str).tolist(),
                                   gdf.geometry.x.round(3).tolist(), gdf.geometry.y.round(3).tolist())))
        parts.append([i["n"] for i in self.iterations])
        return hash(json.dumps(parts, default=str))

    def has_unsaved_changes(self):
        sig = self._state_signature()
        return sig is not None and sig != getattr(self, "_saved_sig", None)

    def _mark_saved(self):
        self._saved_sig = self._state_signature()

    def confirm_discard_changes(self, action):
        """ True when it is fine to go ahead with `action`: nothing unsaved, or the user
        saved or chose to discard. """
        if not self.has_unsaved_changes():
            return True
        box = QMessageBox(self)
        box.setWindowTitle(action)
        box.setIcon(QMessageBox.Warning)
        box.setText("The current workspace has unsaved changes.")
        box.setInformativeText("Save them before continuing?")
        box.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Save)
        choice = box.exec()
        if choice == QMessageBox.Save:
            self.save_workspace_session()
            return not self.has_unsaved_changes()
        return choice == QMessageBox.Discard

    def new_workspace(self):
        if not self.confirm_discard_changes("New Workspace"):
            return
        self._clear_session()
        self._mark_saved()

    def open_workspace_dialog(self):
        """ File > Load Workspace: pick a workspace folder and load it fresh. """
        if not self.confirm_discard_changes("Load Workspace"):
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Workspace Folder", self.workspace_dir or "")
        if not folder:
            return
        if not os.path.exists(os.path.join(folder, "session.json")):
            QMessageBox.warning(self, "Load Workspace", "That folder has no session.json -- it is not a saved workspace.")
            return
        if self.engine.ortho_data is not None:
            self._clear_session()
        self.workspace_dir = folder
        self.load_workspace_session()

    def import_ground_truth_csv(self):
        """ Import > Ground Truth CSV: switches the Accuracy tab to CSV mode and imports. """
        self.acc_gt_csv_radio.setChecked(True)
        self.sync_accuracy_gt_controls()
        self.tabs.setCurrentWidget(self.accuracy_tab)
        self.browse_for_accuracy_csv()

    def save_workspace_as(self):
        """ Copies the orthomosaic, the DSM and the workspace into a chosen folder, saves
        there, and carries on working from the copies. """
        e = self.engine
        if e.ortho_path is None:
            print("Save Workspace As: load an orthomosaic first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Save Workspace As -- choose a folder")
        if not folder:
            return
        folder = os.path.abspath(folder)
        stem = os.path.splitext(os.path.basename(e.ortho_path))[0]
        new_ws = os.path.join(folder, f"{stem}_workspace")
        try:
            new_ortho = self._copy_raster_with_sidecars(e.ortho_path, folder)
            new_dsm = self._copy_raster_with_sidecars(e.dsm_path, folder) if e.dsm_path else None
            if self.workspace_dir and os.path.isdir(self.workspace_dir) and \
                    os.path.abspath(self.workspace_dir) != os.path.abspath(new_ws):
                shutil.copytree(self.workspace_dir, new_ws, dirs_exist_ok=True)
        except Exception as ex:
            QMessageBox.critical(self, "Save Workspace As", f"Copying failed:\n{ex}")
            return
        e.ortho_path = new_ortho
        self.path_edit.setText(new_ortho)
        if new_dsm:
            e.dsm_path = new_dsm
            self.dsm_path_edit.setText(new_dsm)
        self.workspace_dir = new_ws
        self.save_workspace_session()
        print(f"Workspace saved as {new_ws}; now working from the copies in {folder}.")

    def _copy_raster_with_sidecars(self, path, folder):
        """ Copies a raster and its sidecar files (.aux.xml, .ovr, world file) into folder,
        with a progress dialog for large files; returns the new path. """
        from PySide6.QtWidgets import QProgressDialog
        src_dir, name = os.path.split(os.path.abspath(path))
        stem = os.path.splitext(name)[0]
        if os.path.abspath(src_dir) == folder:
            return os.path.abspath(path)
        files = [f for f in os.listdir(src_dir) if f == name or f.startswith(name + ".")
                 or f in (stem + ".tfw", stem + ".tifw", stem + ".wld", stem + ".prj")]
        total = sum(os.path.getsize(os.path.join(src_dir, f)) for f in files)
        dlg = QProgressDialog(f"Copying {name}…", None, 0, 1000, self)
        dlg.setWindowTitle("Save Workspace As")
        dlg.setMinimumDuration(500)
        done = 0
        for f in files:
            with open(os.path.join(src_dir, f), "rb") as fi, open(os.path.join(folder, f), "wb") as fo:
                while True:
                    chunk = fi.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    fo.write(chunk)
                    done += len(chunk)
                    dlg.setValue(int(1000 * done / max(total, 1)))
                    QApplication.processEvents()
            shutil.copystat(os.path.join(src_dir, f), os.path.join(folder, f))
        dlg.setValue(1000)
        print(f" -> Copied {name} ({total / 1e9:.2f} GB) to {folder}")
        return os.path.join(folder, name)

    def closeEvent(self, event):
        if self.confirm_discard_changes("Close Lite Classify"):
            if self.odm_window is not None and self.odm_window.is_running():
                self.odm_window.detach()   # WebODM keeps going; picked up on the next start
            event.accept()
        else:
            event.ignore()

    # --- WebODM (Import > Orthomosaic from Drone Photos) ---
    def _ensure_odm_window(self):
        if self.odm_window is None:
            self.odm_window = OdmWindow(self)
            self.odm_window.status_text.connect(self._on_odm_status)
            self.odm_window.load_requested.connect(self.load_webodm_result)
        return self.odm_window

    def open_odm_window(self):
        w = self._ensure_odm_window()
        w.show()
        w.raise_()
        w.activateWindow()

    def _on_odm_status(self, text):
        self.odm_status_btn.setText(text)
        self.odm_status_btn.setVisible(bool(text))

    def load_webodm_result(self, ortho, dsm):
        """ Loads a finished WebODM orthomosaic and DSM into Tab 1, replacing the current
        workspace after the usual unsaved-changes prompt. """
        if self.engine.ortho_data is not None:
            if not self.confirm_discard_changes("Load WebODM Result"):
                return
            self._clear_session()
        self.tabs.setCurrentIndex(0)
        self.path_edit.setText(ortho)
        self.load_ortho()
        if self.engine.ortho_data is not None and os.path.exists(dsm):
            self.dsm_path_edit.setText(dsm)
            self.load_dsm()
        self.raise_()
        self.activateWindow()

    # --- TAB 1: Load, View & Segment (merged Load&View + Segmentation) ---
    def init_load_view_tab(self):
        """ Merged Load/View + Segmentation controls into a single tab -- both control
        columns were short enough to fit one scrollable column, and this removes a
        click from the load -> segment -> classify flow. """
        tab1 = QWidget()
        layout = QHBoxLayout(tab1)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        scroll_area.setMaximumWidth(370)

        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        scroll_area.setWidget(controls_widget)


        # --- Input Data: Ortho + DSM merged into one group, auto-loads on selection ---
        input_group = QGroupBox("Input Data")
        input_layout = QVBoxLayout()

        input_layout.addWidget(QLabel("Orthomosaic:"))
        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Path to orthomosaic...")
        self.path_edit.returnPressed.connect(self.load_ortho)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_for_ortho)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(browse_btn)
        input_layout.addLayout(path_layout)
        webodm_link = QLabel('<a href="#">No orthomosaic yet? Process drone photos…</a>')
        webodm_link.setToolTip("Import > Orthomosaic from Drone Photos (WebODM)")
        webodm_link.linkActivated.connect(lambda _: self.open_odm_window())
        input_layout.addWidget(webodm_link)

        input_layout.addWidget(QLabel("Elevation Model (Optional DSM):"))
        dsm_path_lay = QHBoxLayout()
        self.dsm_path_edit = QLineEdit()
        self.dsm_path_edit.setPlaceholderText("Path to matching DSM...")
        self.dsm_path_edit.returnPressed.connect(self.load_dsm)
        dsm_browse_btn = QPushButton("Browse")
        dsm_browse_btn.clicked.connect(self.browse_for_dsm)
        dsm_path_lay.addWidget(self.dsm_path_edit)
        dsm_path_lay.addWidget(dsm_browse_btn)
        input_layout.addLayout(dsm_path_lay)

        unload_dsm_tab1_btn = QPushButton("Unload DSM")
        unload_dsm_tab1_btn.clicked.connect(self.unload_surface_data)
        input_layout.addWidget(unload_dsm_tab1_btn)

        # --- Resample Orthomosaic: reload from disk at a coarser/finer resolution,
        # so a huge photogrammetry export can be worked with in-app instead of round-
        # tripping back to the photogrammetry/GIS software just to downsample it. ---
        input_layout.addWidget(QLabel("Resample Scale (% of native resolution):"))
        resample_row = QHBoxLayout()
        self.resample_scale_slider = QSlider(Qt.Horizontal)
        self.resample_scale_slider.setRange(5, 100)
        self.resample_scale_slider.setValue(100)
        self.resample_scale_spinbox = QSpinBox()
        self.resample_scale_spinbox.setRange(5, 100)
        self.resample_scale_spinbox.setValue(100)
        self.resample_scale_spinbox.setSuffix("%")
        self.resample_scale_slider.valueChanged.connect(self.resample_scale_spinbox.setValue)
        self.resample_scale_spinbox.valueChanged.connect(self.resample_scale_slider.setValue)
        self.resample_scale_slider.valueChanged.connect(self._on_resample_scale_changed)
        resample_row.addWidget(self.resample_scale_slider)
        resample_row.addWidget(self.resample_scale_spinbox)
        input_layout.addLayout(resample_row)

        # Absolute target-resolution text input -- only shown for a projected,
        # linear-unit source CRS (e.g. UTM, meters/px), where "cm/px" is a fixed
        # physical quantity. Hidden for a geographic (lat/lon degree) source, where an
        # absolute cm value isn't meaningful without a latitude-dependent scale factor
        # -- the percent slider above still works for those. Hidden by default until a
        # loaded file's CRS is known.
        target_res_row = QHBoxLayout()
        self.resample_target_res_label = QLabel("Target Resolution:")
        target_res_row.addWidget(self.resample_target_res_label)
        self.resample_target_res_spinbox = QDoubleSpinBox()
        self.resample_target_res_spinbox.setDecimals(2)
        self.resample_target_res_spinbox.setSingleStep(0.5)
        self.resample_target_res_spinbox.setSuffix(" cm/px")
        self.resample_target_res_spinbox.valueChanged.connect(self._on_resample_target_res_changed)
        target_res_row.addWidget(self.resample_target_res_spinbox)
        input_layout.addLayout(target_res_row)
        self.resample_target_res_label.setVisible(False)
        self.resample_target_res_spinbox.setVisible(False)

        self.resample_preview_label = QLabel("Load an orthomosaic to see its native resolution.")
        self.resample_preview_label.setStyleSheet("color: #888; font-size: 11px;")
        self.resample_preview_label.setWordWrap(True)
        input_layout.addWidget(self.resample_preview_label)

        self.apply_resample_btn = QPushButton("Apply Resample")
        self.apply_resample_btn.setEnabled(False)
        self.apply_resample_btn.setToolTip(
            "Reloads the orthomosaic from disk at the chosen scale. Clears the current "
            "segmentation, DSM linkage, and classification result (training samples too) "
            "since they're tied to the old pixel grid -- your class list and colors stay."
        )
        self.apply_resample_btn.clicked.connect(self.apply_resample)
        input_layout.addWidget(self.apply_resample_btn)

        input_group.setLayout(input_layout)
        controls_layout.addWidget(input_group)

        # --- Color Correction: collapsible, starts collapsed ---
        color_content = QWidget()
        color_layout = QFormLayout(color_content)
        color_layout.setContentsMargins(0, 4, 0, 0)

        def create_slider_spinbox(range_min, range_max, initial_val, decimals, step):
            slider = QSlider(Qt.Horizontal)
            slider.setRange(range_min, range_max)
            slider.setValue(initial_val)
            spinbox = QDoubleSpinBox()
            spinbox.setRange(range_min / (10**decimals), range_max / (10**decimals))
            spinbox.setValue(initial_val / (10**decimals))
            spinbox.setDecimals(decimals)
            spinbox.setSingleStep(step)

            slider.valueChanged.connect(lambda val: spinbox.setValue(val / (10**decimals)))
            spinbox.valueChanged.connect(lambda val: slider.setValue(int(val * (10**decimals))))
            slider.sliderReleased.connect(self.apply_color_correction)
            spinbox.editingFinished.connect(self.apply_color_correction)

            container = QHBoxLayout()
            container.addWidget(slider)
            container.addWidget(spinbox)
            return container, slider, spinbox

        brightness_layout, self.brightness_slider, self.brightness_spinbox = create_slider_spinbox(-100, 100, 0, 2, 0.01)
        color_layout.addRow("Brightness:", brightness_layout)
        contrast_layout, self.contrast_slider, self.contrast_spinbox = create_slider_spinbox(0, 200, 100, 2, 0.01)
        color_layout.addRow("Contrast:", contrast_layout)
        hue_layout, self.hue_slider, self.hue_spinbox = create_slider_spinbox(-180, 180, 0, 1, 0.1)
        color_layout.addRow("Hue:", hue_layout)
        saturation_layout, self.saturation_slider, self.saturation_spinbox = create_slider_spinbox(0, 200, 100, 2, 0.01)
        color_layout.addRow("Saturation:", saturation_layout)
        temp_layout, self.temperature_slider, self.temperature_spinbox = create_slider_spinbox(-100, 100, 0, 2, 0.01)
        color_layout.addRow("Temperature:", temp_layout)

        red_layout, self.red_slider, self.red_spinbox = create_slider_spinbox(0, 200, 100, 2, 0.01)
        color_layout.addRow("Red:", red_layout)
        green_layout, self.green_slider, self.green_spinbox = create_slider_spinbox(0, 200, 100, 2, 0.01)
        color_layout.addRow("Green:", green_layout)
        blue_layout, self.blue_slider, self.blue_spinbox = create_slider_spinbox(0, 200, 100, 2, 0.01)
        color_layout.addRow("Blue:", blue_layout)

        auto_correct_btn = QPushButton("Auto-Correct")
        auto_correct_btn.clicked.connect(self.auto_correct_colors)
        reset_colors_btn = QPushButton("Reset Colors")
        reset_colors_btn.clicked.connect(self.reset_colors)

        button_layout = QHBoxLayout()
        button_layout.addWidget(auto_correct_btn)
        button_layout.addWidget(reset_colors_btn)
        color_layout.addRow(button_layout)

        color_section = CollapsibleSection("Color Correction", color_content, start_collapsed=True)
        controls_layout.addWidget(color_section)

        # --- SLIC Segmentation Parameters ---
        slic_group = QGroupBox("SLIC Segmentation Parameters")
        slic_layout = QFormLayout()

        self.n_segments_slider = QSlider(Qt.Horizontal)
        self.n_segments_slider.setRange(100, 200000)
        self.n_segments_slider.setValue(5000)
        self.n_segments_spinbox = QSpinBox()
        self.n_segments_spinbox.setRange(100, 2000000)   # large scenes need more than the slider's range
        self.n_segments_spinbox.setGroupSeparatorShown(True)
        self.n_segments_spinbox.setValue(5000)

        # --- Setting up 500 step snaps on segmentation slider ---
        def snap_segments_slider(value):
            snapped = round(value / 500.0) * 500
            snapped = max(100, min(200000, snapped))
            self.n_segments_slider.blockSignals(True)
            self.n_segments_slider.setValue(snapped)
            self.n_segments_slider.blockSignals(False)
            self.n_segments_spinbox.setValue(snapped)

        self.n_segments_slider.valueChanged.connect(snap_segments_slider)
        def spinbox_to_slider(value):
            # quietly: a count above the slider's range must not be clamped back by it
            self.n_segments_slider.blockSignals(True)
            self.n_segments_slider.setValue(value)
            self.n_segments_slider.blockSignals(False)
        self.n_segments_spinbox.valueChanged.connect(spinbox_to_slider)

        # Object size on the ground, linked to the count both ways (count = valid area /
        # size). Whichever was edited last drives the other, so a resample or a new image
        # keeps the size the user asked for. Projected CRS only (see _sync_segment_size).
        self.seg_size_spinbox = QDoubleSpinBox()
        self.seg_size_spinbox.setRange(0.5, 10000.0)
        self.seg_size_spinbox.setDecimals(1)
        self.seg_size_spinbox.setSingleStep(1.0)
        self.seg_size_spinbox.setSuffix(" m²")
        self.seg_size_spinbox.setValue(self.DEFAULT_SEGMENT_SIZE_M2)
        self.seg_size_spinbox.setToolTip(
            "Target size of one segment on the ground. Sets Num Segments from the image's valid "
            "area. Crown- and roof-part objects of a few to a few tens of m² suit most surveys; "
            "make it small enough that the narrowest thing you map gets segments of its own.")
        self._seg_size_drives = True    # the size box drives the count until the count is edited
        self._seg_syncing = False
        self.n_segments_spinbox.valueChanged.connect(self._segment_count_edited)
        self.seg_size_spinbox.valueChanged.connect(self._segment_size_edited)
        self.n_segments_spinbox.valueChanged.connect(self.update_segmentation_plan_label)

        self.seg_size_label = QLabel("Object Size:")
        self.seg_size_label.setVisible(False)      # shown once an image in projected coordinates is loaded
        self.seg_size_spinbox.setVisible(False)
        slic_layout.addRow(self.seg_size_label, self.seg_size_spinbox)
        slic_layout.addRow("Num Segments:", self.n_segments_slider)
        slic_layout.addRow("", self.n_segments_spinbox)
        self.seg_plan_label = QLabel("")
        self.seg_plan_label.setStyleSheet("color: #888; font-size: 11px;")
        self.seg_plan_label.setWordWrap(True)
        slic_layout.addRow("", self.seg_plan_label)

        self.compactness_slider = QSlider(Qt.Horizontal)
        self.compactness_slider.setRange(1, 100)
        self.compactness_slider.setValue(15)
        self.compactness_dspinbox = QDoubleSpinBox()
        self.compactness_dspinbox.setRange(0.01, 1.0)
        self.compactness_dspinbox.setSingleStep(0.01)
        self.compactness_dspinbox.setValue(0.15)
        self.compactness_slider.valueChanged.connect(lambda val: self.compactness_dspinbox.setValue(val / 100.0))
        self.compactness_dspinbox.valueChanged.connect(lambda val: self.compactness_slider.setValue(int(val * 100)))
        slic_layout.addRow("Compactness:", self.compactness_slider)
        slic_layout.addRow("", self.compactness_dspinbox)

        self.sigma_dspinbox = QDoubleSpinBox()
        self.sigma_dspinbox.setRange(0.0, 10.0)
        self.sigma_dspinbox.setSingleStep(0.1)
        self.sigma_dspinbox.setValue(2.5)
        slic_layout.addRow("Sigma (Smoothing):", self.sigma_dspinbox)

        # Colour/texture can be measured finer than the segmentation grid, straight from the
        # source file (see ObiaEngine._compute_features_from_file). Projected CRS only.
        self.feature_res_label = QLabel("Colour/texture at:")
        self.feature_res_spinbox = QDoubleSpinBox()
        self.feature_res_spinbox.setDecimals(2)
        self.feature_res_spinbox.setSingleStep(0.5)
        self.feature_res_spinbox.setSuffix(" cm/px")
        self.feature_res_spinbox.setToolTip(
            "Resolution at which each segment's colour and texture are measured. The default is "
            "the segmentation (working) resolution. A finer value reads the original file block by "
            "block, so segments can be drawn on a coarse grid while their colour and texture "
            "come from the full detail. Texture then describes finer detail. Takes effect at the "
            "next Run Segmentation.")
        self.feature_res_spinbox.valueChanged.connect(self.update_segmentation_plan_label)
        slic_layout.addRow(self.feature_res_label, self.feature_res_spinbox)
        self.feature_res_label.setVisible(False)
        self.feature_res_spinbox.setVisible(False)
        slic_group.setLayout(slic_layout)
        controls_layout.addWidget(slic_group)

        # --- Segmentation Engine: GPU/CPU toggle ---
        engine_group = QGroupBox("Segmentation Engine")
        engine_layout = QHBoxLayout()
        self.cpu_radio = QRadioButton("CPU")
        self.gpu_radio = QRadioButton("GPU")
        self.engine_btn_group = QButtonGroup(self)
        self.engine_btn_group.addButton(self.cpu_radio)
        self.engine_btn_group.addButton(self.gpu_radio)

        gpu_available = cuda_available()
        self.gpu_radio.setEnabled(gpu_available)
        if not gpu_available:
            self.gpu_radio.setToolTip("No CUDA-capable GPU detected on this machine.")
        (self.gpu_radio if gpu_available else self.cpu_radio).setChecked(True)

        engine_layout.addWidget(self.cpu_radio)
        engine_layout.addWidget(self.gpu_radio)
        self.gpu_radio.toggled.connect(self.update_segmentation_plan_label)
        engine_group.setLayout(engine_layout)
        controls_layout.addWidget(engine_group)

        # --- Execution & Storage (Save Workspace lives on the Classification tab now) ---
        io_group = QGroupBox("Execution & Storage")
        io_layout = QVBoxLayout()
        self.run_seg_btn = QPushButton("Run Segmentation")
        self.run_seg_btn.clicked.connect(self.run_segmentation)
        io_layout.addWidget(self.run_seg_btn)
        self.cancel_seg_btn = QPushButton("Cancel")
        self.cancel_seg_btn.setToolTip("Stop segmenting and keep the previous result. Takes effect at "
                                       "the next checkpoint (between tiles, or during feature measurement).")
        self.cancel_seg_btn.clicked.connect(self.cancel_segmentation)
        self.cancel_seg_btn.setVisible(False)
        io_layout.addWidget(self.cancel_seg_btn)
        io_group.setLayout(io_layout)
        controls_layout.addWidget(io_group)

        controls_layout.addStretch()

        # --- Shared viewer with boundary + elevation overlay toggles ---
        viewer_container = QWidget()
        viewer_layout = QVBoxLayout(viewer_container)
        viewer_layout.setContentsMargins(0, 0, 0, 0)

        overlay_bar = QHBoxLayout()
        self.show_boundaries_check = QCheckBox("Show Segment Boundaries")
        self.show_boundaries_check.setEnabled(False)
        self.show_boundaries_check.stateChanged.connect(self.refresh_load_view_overlay)
        overlay_bar.addWidget(self.show_boundaries_check)

        self.show_elevation_check = QCheckBox("Show Elevation")
        self.show_elevation_check.setEnabled(False)
        self.show_elevation_check.stateChanged.connect(self.on_elevation_toggle_changed)
        overlay_bar.addWidget(self.show_elevation_check)

        overlay_bar.addWidget(QLabel("Opacity:"))
        self.elevation_opacity_slider = QSlider(Qt.Horizontal)
        self.elevation_opacity_slider.setRange(0, 100)
        self.elevation_opacity_slider.setValue(60)
        self.elevation_opacity_slider.setEnabled(False)
        self.elevation_opacity_slider.setMaximumWidth(120)
        self.elevation_opacity_slider.valueChanged.connect(self.refresh_load_view_overlay)
        overlay_bar.addWidget(self.elevation_opacity_slider)
        overlay_bar.addWidget(self._new_map_info(), 1)
        viewer_layout.addWidget(self._map_header([overlay_bar]))

        self.viewer = InteractivePhotoViewer()
        self.viewer.set_detail_provider(self._detail_image)
        viewer_layout.addWidget(self.viewer, 1)
        viewer_layout.addWidget(self._map_tip("Scroll to zoom, drag to pan. The view is shared with the other tabs."))

        layout.addWidget(scroll_area)
        layout.addWidget(viewer_container)
        self.tabs.addTab(tab1, "1. Load, View & Segment")

    def browse_for_ortho(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Orthomosaic", "", "Image Files (*.tif *.tiff)")
        if file_path:
            self.path_edit.setText(file_path)
            self.load_ortho()

    def browse_for_dsm(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Digital Surface Model (DSM)", "", "Raster files (*.tif *.tiff)")
        if file_path:
            self.dsm_path_edit.setText(file_path)
            self.load_dsm()

    def load_ortho(self, out_shape=None):
        filepath = self.path_edit.text()
        if not os.path.exists(filepath):
            print("Error: Input file path not found.")
            return
        try:
            print(f"Loading image from: {filepath}...")
            QApplication.processEvents()
            try:
                self.engine.load_raster(filepath, out_shape=out_shape)
            finally:
                # Whatever happened, the resample controls must describe the file now
                # loaded (or none) and stay usable -- a failed load used to leave Apply
                # Resample greyed out and the Target Resolution row hidden.
                self._configure_resample_controls_for_new_source()
            if out_shape is None:   # a workspace grid is chosen, not forced by size
                self._report_reduced_load()

            base_dir, file_name = os.path.split(filepath)
            base_name, _ = os.path.splitext(file_name)
            self.workspace_dir = os.path.join(base_dir, f"{base_name}_workspace")
            # Not created here -- an empty <name>_workspace folder shouldn't appear
            # next to a tif just because it was loaded/browsed to (including while
            # relocating a moved raster during Load Workspace, which immediately
            # overwrites this path anyway). It's created lazily on an actual save --
            # see _ensure_workspace_dir().
            print(f"Active workspace synced: {self.workspace_dir}")

            self.reset_colors()
            self._sync_segment_size()
            self._sync_mmu_m2()
            self.update_segmentation_plan_label()

            # Show the whole image on import -- otherwise the view stays at whatever
            # zoom/pan the previous image (if any) happened to leave it at, which for
            # a large orthomosaic can mean a near-blank, heavily-zoomed-in crop.
            self.viewer.fit_view()
            self.acc_viewer.fit_view()

            print("Raster successfully loaded.")
        except Exception as e:
            print(f"Load failed: {e}")

    def _configure_resample_controls_for_new_source(self):
        """ Called once per freshly loaded orthomosaic file (not on every resample of
        the same file) -- resets the resample controls to "100% / native resolution"
        and shows the absolute Target Resolution field only when the source has a
        projected, linear-unit CRS (e.g. UTM), since an absolute cm/px figure isn't a
        fixed physical quantity for a geographic (lat/lon degree) CRS without a
        latitude-dependent scale factor. The percent slider still works either way. """
        show_target_res = (self.engine.native_pixel_size is not None) and (not self.engine.native_crs_is_geographic)
        self.resample_target_res_label.setVisible(show_target_res)
        self.resample_target_res_spinbox.setVisible(show_target_res)

        # Start from the resolution actually loaded (native, or reduced for a huge file).
        eff = self.engine.effective_pixel_size or self.engine.native_pixel_size
        scale = 100
        if eff and self.engine.native_pixel_size:
            scale = max(5, min(100, int(round(100.0 * self.engine.native_pixel_size / eff))))
        self._syncing_resample_controls = True
        self.resample_scale_slider.setValue(scale)
        self.resample_scale_spinbox.setValue(scale)
        if show_target_res:
            native_cm = self.engine.native_pixel_size * 100.0
            # Floor = native (100%, can't go finer via this control -- upsampling
            # doesn't add real detail); ceiling = 20x coarser, matching the slider's
            # 5% floor.
            self.resample_target_res_spinbox.setRange(native_cm, native_cm * 20.0)
            self.resample_target_res_spinbox.setValue(eff * 100.0)
        self._syncing_resample_controls = False

        self.apply_resample_btn.setEnabled(self.engine.native_pixel_size is not None)
        self.update_resample_preview_label()
        self._configure_feature_res_control()

    def _on_resample_scale_changed(self, value):
        """ Percent slider moved -- push the equivalent absolute resolution into the
        Target Resolution field (for a projected CRS) so the two stay in sync, then
        refresh the preview. Guarded against re-entrancy from
        _on_resample_target_res_changed's own reverse sync. """
        if not self._syncing_resample_controls and self.engine.native_pixel_size is not None and not self.engine.native_crs_is_geographic:
            native_cm = self.engine.native_pixel_size * 100.0
            self._syncing_resample_controls = True
            self.resample_target_res_spinbox.setValue(native_cm * (100.0 / value))
            self._syncing_resample_controls = False
        self.update_resample_preview_label()

    def _on_resample_target_res_changed(self, value):
        """ Target Resolution field edited directly -- this becomes the authoritative
        value (apply_resample reads it precisely, not rounded), and the percent slider
        is nudged to the nearest matching percent purely as a visual cue. Only wired up
        for a projected CRS; for a geographic source the field stays hidden and this is
        never triggered by user input. """
        if self._syncing_resample_controls or self.engine.native_pixel_size is None or self.engine.native_crs_is_geographic:
            return
        if value <= 0:
            return
        native_cm = self.engine.native_pixel_size * 100.0
        scale = max(5, min(100, int(round(100.0 * native_cm / value))))
        self._syncing_resample_controls = True
        self.resample_scale_slider.setValue(scale)
        self.resample_scale_spinbox.setValue(scale)
        self._syncing_resample_controls = False
        self.update_resample_preview_label()

    def _resample_target_factor(self):
        """ The downsample factor (>= 1.0, native / target) implied by whichever
        control is authoritative for this source's CRS -- the precise Target
        Resolution field for a projected CRS, the percent slider for a geographic one.
        Returns None if no orthomosaic is loaded yet. """
        if self.engine.native_pixel_size is None:
            return None
        if self.engine.native_crs_is_geographic:
            return 100.0 / self.resample_scale_slider.value()
        native_cm = self.engine.native_pixel_size * 100.0
        target_cm = self.resample_target_res_spinbox.value()
        if target_cm <= 0:
            return None
        return target_cm / native_cm

    def update_resample_preview_label(self):
        if self.engine.native_pixel_size is None:
            self.resample_preview_label.setText("Load an orthomosaic to see its native resolution.")
            return

        factor = self._resample_target_factor() or 1.0
        out_w = max(1, round(self.engine.native_width / factor))
        out_h = max(1, round(self.engine.native_height / factor))

        if self.engine.native_crs_is_geographic:
            unit, native_val, target_val = "deg/px", self.engine.native_pixel_size, self.engine.native_pixel_size * factor
        else:
            unit, native_val, target_val = "cm/px", self.engine.native_pixel_size * 100.0, self.engine.native_pixel_size * factor * 100.0

        if factor <= 1.0001:
            self.resample_preview_label.setText(
                f"Native resolution: {native_val:.3f} {unit} "
                f"({self.engine.native_width}×{self.engine.native_height} px)."
            )
        else:
            self.resample_preview_label.setText(
                f"Native: {native_val:.3f} {unit} ({self.engine.native_width}×{self.engine.native_height} px)  "
                f"→  target: {target_val:.3f} {unit} ({out_w}×{out_h} px)."
            )

    def apply_resample(self):
        if self.engine.ortho_path is None:
            print("Error: Load an orthomosaic first before resampling.")
            return

        use_target_res = not self.engine.native_crs_is_geographic
        if use_target_res:
            target_desc = f"{self.resample_target_res_spinbox.value():.2f} cm/px"
        else:
            target_desc = f"{self.resample_scale_slider.value()}% of native resolution"

        has_downstream_work = (
            self.engine.segmented_image is not None
            or self.classified_image_data is not None
            or any(data["training_segments"] for data in self.classes.values())
        )
        if has_downstream_work:
            reply = QMessageBox.question(
                self, "Resample Orthomosaic",
                f"Resample the orthomosaic to {target_desc}? "
                "This reloads the image from disk at the new resolution and clears the "
                "current segmentation, DSM linkage, and any classification result. Your "
                "class list and colors are kept, but existing training samples will be "
                "cleared since they reference segments that won't exist after resampling. "
                "Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        if not self._confirm_working_size(target_desc):
            return

        had_dsm = bool(self.dsm_path_edit.text()) and os.path.exists(self.dsm_path_edit.text())

        try:
            print(f"Resampling orthomosaic to {target_desc} (reloading from disk)...")
            QApplication.processEvents()
            if use_target_res:
                self.engine.resample_ortho(target_pixel_size=self.resample_target_res_spinbox.value() / 100.0)
            else:
                self.engine.resample_ortho(scale_percent=self.resample_scale_slider.value())

            w, h = self.engine.ortho_data.shape[2], self.engine.ortho_data.shape[1]
            unit = "deg/px" if self.engine.native_crs_is_geographic else "cm/px"
            eff = self.engine.effective_pixel_size if self.engine.native_crs_is_geographic else self.engine.effective_pixel_size * 100.0
            print(f"Resampled to {w}x{h} px (~{eff:.3f} {unit}).")

            # Segmentation/classification were computed against the previous pixel
            # grid -- none of it carries over to the new one (see resample_ortho's
            # docstring for why this can't be preserved).
            for data in self.classes.values():
                data["training_segments"] = set()
            self.classifier = None
            self.classified_image_data = None
            self.raw_classified_raster_ints = None
            self.sieved_classified_raster_ints = None
            self.base_seg_qimg = None
            self.boundary_overlay_qimg = None
            self.sample_class_map = None
            self.selected_segments = set()
            self.locked_selection = set()
            self.current_seed_segments = set()
            self.selection_undo_stack = []
            self.selection_count_label.setText("Selected: 0 segments")
            self.show_boundaries_check.setChecked(False)
            self.show_boundaries_check.setEnabled(False)
            self.current_view_mode = "segmented"
            self.seg_view_btn.setChecked(True)
            self.cls_view_btn.setChecked(False)
            self.sync_classified_button_label()
            self._sync_blend_availability()
            self.update_list_widget_counts()
            self.class_viewer.set_image(QImage())

            self.reset_colors()
            self.update_resample_preview_label()
            self._configure_feature_res_control()
            self._sync_segment_size()
            self._sync_mmu_m2()
            self.update_segmentation_plan_label()

            # The image's pixel dimensions just changed -- show the whole thing rather
            # than leaving the view at a zoom/pan level computed for the old size.
            # (Accuracy points are stored in real-world CRS coordinates, not pixel
            # indices, so they don't need clearing here -- they reproject onto the new
            # grid automatically wherever they're drawn/hit-tested.)
            self.viewer.fit_view()
            self.acc_viewer.fit_view()

            if had_dsm:
                print("Reloading DSM to match the new orthomosaic resolution...")
                self.load_dsm()
            else:
                self.update_dsm_status_label()
                self.rebuild_hillshade_cache()

            print("Resample complete. Re-run segmentation to continue.")
        except Exception as e:
            print(f"Resample failed: {e}")

    def load_dsm(self):
        filepath = self.dsm_path_edit.text()
        if not os.path.exists(filepath):
            print("Error: DSM file path not found.")
            return
        if self.engine.ortho_data is None:
            print("Error: Load your base orthomosaic map first before processing structural elevation.")
            return
        try:
            desc = self.engine.load_dsm(filepath)
            print(f"DSM attached ({desc}). It is matched to the working resolution when elevation is "
                  f"first needed (at segmentation), so resample the orthomosaic first if you plan to.")
            self.update_dsm_status_label()
            self.rebuild_hillshade_cache()
        except Exception as e:
            print(f"DSM could not be loaded: {e}")

    def rebuild_hillshade_cache(self):
        """ Rebuilds the colorized hillshade overlay from engine.dsm_data and
        enables/disables the "Show Elevation" toggle accordingly. Called after any
        DSM load/reload/unload so the toggle's availability always matches reality. """
        rgba = self.engine.compute_hillshade_rgba(max_side=2048)  # small preview, stretched when drawn
        if rgba is None:
            self.hillshade_qimg = None
            self.show_elevation_check.setChecked(False)
            self.show_elevation_check.setEnabled(False)
            self.elevation_opacity_slider.setEnabled(False)
            self.refresh_load_view_overlay()
            return
        rgba_contig = np.ascontiguousarray(rgba)
        h, w, ch = rgba_contig.shape
        self.hillshade_qimg = QImage(rgba_contig.data, w, h, ch * w, QImage.Format_RGBA8888).copy()
        self.show_elevation_check.setEnabled(True)
        self.refresh_load_view_overlay()

    def on_elevation_toggle_changed(self):
        self.elevation_opacity_slider.setEnabled(self.show_elevation_check.isChecked())
        self.refresh_load_view_overlay()

    def refresh_load_view_overlay(self):
        """ Composites the optional boundary + elevation layers on top of the current
        base_raw_qimg and pushes the result to self.viewer -- the merged tab's
        equivalent of update_classification_display's "raw" branch (draw the base
        image, then optional overlays on top with QPainter), just with independent
        toggles instead of a single view-mode selector. """
        if self.base_raw_qimg is None:
            return
        # Overlays are separate layers in the viewer (no full-size copy of the ortho).
        self.viewer.set_base_image_cached(self.raw_pixmap, grid_size=self.raw_grid)
        layers = []
        if self.show_elevation_check.isChecked() and self.hillshade_qimg is not None:
            layers.append((self.hillshade_qimg, self.elevation_opacity_slider.value() / 100.0))
        if self.show_boundaries_check.isChecked() and self.boundary_overlay_qimg is not None:
            layers.append((self.boundary_overlay_qimg, 1.0))
        self.viewer.set_overlays(layers)

    def auto_correct_colors(self):
        if self.engine.original_ortho_data is None: return
        print("Applying auto-correction...")
        p2, p98 = np.percentile(self.engine.original_ortho_data, (2, 98))
        self.engine.ortho_data = exposure.rescale_intensity(self.engine.original_ortho_data, in_range=(p2, p98))
        self.reset_sliders()
        self.update_display_from_data()

    def reset_colors(self):
        if self.engine.original_ortho_data is None: return
        self.engine.ortho_data = self.engine.original_ortho_data   # nothing writes into it in place
        self.reset_sliders()
        self.update_display_from_data()

    def reset_sliders(self):
        self.brightness_spinbox.setValue(0)
        self.contrast_spinbox.setValue(1.0)
        self.hue_spinbox.setValue(0)
        self.saturation_spinbox.setValue(1.0)
        self.temperature_spinbox.setValue(0)
        self.red_spinbox.setValue(1.0)
        self.green_spinbox.setValue(1.0)
        self.blue_spinbox.setValue(1.0)

    def apply_color_correction(self):
        if self.engine.original_ortho_data is None: return
        image = np.transpose(self.engine.original_ortho_data, (1, 2, 0)).astype(np.float32) / 255.0

        brightness = self.brightness_spinbox.value()
        contrast = self.contrast_spinbox.value()
        saturation = self.saturation_spinbox.value()
        hue = self.hue_spinbox.value() / 360.0
        temperature = self.temperature_spinbox.value()
        red_gain = self.red_spinbox.value()
        green_gain = self.green_spinbox.value()
        blue_gain = self.blue_spinbox.value()

        image[:, :, 0] *= red_gain
        image[:, :, 1] *= green_gain
        image[:, :, 2] *= blue_gain

        image[:, :, 0] += temperature
        image[:, :, 2] -= temperature

        image = image + brightness
        image = (image - 0.5) * contrast + 0.5

        hsv = rgb2hsv(np.clip(image, 0, 1))
        hsv[:, :, 0] = (hsv[:, :, 0] + hue) % 1.0
        hsv[:, :, 1] *= saturation
        image = hsv2rgb(hsv)
        image = np.clip(image, 0, 1)

        self.engine.ortho_data = (np.transpose(image, (2, 0, 1)) * 255).astype(self.engine.original_ortho_data.dtype)
        self.update_display_from_data()

    def update_display_from_data(self):
        """ Builds the base ortho image once, as RGB32 so its pixmap shares the same memory,
        and hands that one pixmap to every map view (Tab 1, Discover, Classification,
        Accuracy) -- previously each view converted its own full-size copy. """
        if self.engine.ortho_data is None: return
        d = self.engine.ortho_data
        gh, gw = d.shape[1:]
        self.raw_grid = (gw, gh)
        # A very large image is shown from a reduced copy (stretched over the grid); the
        # viewers draw the visible area sharp on top of it when zoomed in (_detail_image).
        st = max(1, int(math.ceil(math.sqrt(gw * gh / DISPLAY_BASE_MAX_PX))))
        if st > 1:
            d = d[:, ::st, ::st]
        if d.dtype != np.uint8:
            d = ((d / d.max()) * 255).astype(np.uint8)
        h, w = d.shape[1:]
        buf = np.empty((h, w, 4), dtype=np.uint8)
        buf[..., 0], buf[..., 1], buf[..., 2], buf[..., 3] = d[2], d[1], d[0], 255
        self.base_raw_qimg = QImage(buf.data, w, h, 4 * w, QImage.Format_RGB32).copy()
        del buf
        self.raw_display_rgb = None
        self.raw_pixmap = QPixmap.fromImage(self.base_raw_qimg)
        self.refresh_load_view_overlay()
        self.update_accuracy_map_display()

    def _detail_image(self, x0, y0, x1, y1, step):
        """ Detail provider for the viewers: the working-grid ortho (as currently colour
        corrected) for grid pixels [y0:y1:step, x0:x1:step], as an RGB32 QImage. """
        d = self.engine.ortho_data
        if d is None:
            return None
        crop = d[:, y0:y1:step, x0:x1:step]
        if crop.size == 0:
            return None
        if crop.dtype != np.uint8:
            crop = ((crop / max(1, d.max())) * 255).astype(np.uint8)
        return bgrx_qimage(np.transpose(crop, (1, 2, 0)))

    def _report_reduced_load(self):
        e = self.engine
        f = getattr(e, "load_reduction", 1.0)
        if f <= 1.001 or e.native_pixel_size is None:
            return
        h, w = e.ortho_data.shape[1:]
        need = working_bytes(e.native_height, e.native_width) / 1e9
        if e.native_crs_is_geographic:
            res = f"{e.effective_pixel_size:.6f} deg/px"
        else:
            res = f"{e.effective_pixel_size * 100:.2f} cm/px (native {e.native_pixel_size * 100:.2f})"
        print(f"Large image: {e.native_width}x{e.native_height} px would need about {need:.0f} GB at native "
              f"resolution, so it was read at {res}, {w}x{h} px. Choose the working resolution under "
              f"Target Resolution and press Apply Resample. Colour and texture can still be measured "
              f"finer than that (SLIC Segmentation Parameters > Colour/texture at).")

    def _working_size_for(self, factor):
        e = self.engine
        return (max(1, round(e.native_height / factor)), max(1, round(e.native_width / factor)))

    def _confirm_working_size(self, target_desc):
        """ Before a resample: will the image at that resolution (plus what segmentation
        keeps per pixel) fit in memory? Warns with a suggested resolution if not. """
        factor = self._resample_target_factor()
        if not factor:
            return True
        h, w = self._working_size_for(factor)
        has_dsm = self.engine.has_dsm
        need = working_bytes(h, w, has_dsm)
        ram = free_ram_bytes()
        if ram is None:
            return True
        held = 0 if self.engine.ortho_data is None else working_bytes(*self.engine.ortho_data.shape[1:], has_dsm)
        avail = ram + held * 0.3   # part of what the current image holds is freed first
        if need <= 0.7 * avail:
            return True
        per_px = need / (h * w)
        ok_px = 0.6 * avail / per_px
        suggest = self.engine.native_pixel_size * float(np.sqrt(self.engine.native_width * self.engine.native_height / ok_px))
        unit_val = f"{suggest * 100:.1f} cm/px" if not self.engine.native_crs_is_geographic else f"{suggest:.6f} deg/px"
        reply = QMessageBox.warning(
            self, "Not Enough Memory",
            f"At {target_desc} the image is {w:,} x {h:,} px. Working on it (image, segment map"
            f"{', DSM' if has_dsm else ''}) needs about {need / 1e9:.1f} GB, but about {avail / 1e9:.1f} GB "
            f"is available.\n\nSuggested: {unit_val} or coarser. Colour and texture can still be "
            f"measured at a finer resolution (Colour/texture at).\n\nResample anyway?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return reply == QMessageBox.Yes

    def _configure_feature_res_control(self):
        """ Range: native (finest the file has) .. the working grid; default the working
        grid, i.e. colour/texture measured where the segments are drawn. """
        e = self.engine
        show = (e.native_pixel_size is not None and not e.native_crs_is_geographic
                and e.effective_pixel_size is not None)
        self.feature_res_label.setVisible(show)
        self.feature_res_spinbox.setVisible(show)
        if not show:
            return
        native_cm, work_cm = e.native_pixel_size * 100.0, e.effective_pixel_size * 100.0
        self.feature_res_spinbox.blockSignals(True)
        self.feature_res_spinbox.setRange(native_cm, max(native_cm, work_cm))
        self.feature_res_spinbox.setValue(work_cm)
        self.feature_res_spinbox.blockSignals(False)
        self.feature_res_spinbox.setEnabled(work_cm > native_cm * 1.05)

    def _chosen_feature_pixel_size(self):
        """ The Colour/texture-at value in CRS units, or None for the working grid. """
        e = self.engine
        if self.feature_res_spinbox.isHidden() or e.effective_pixel_size is None:
            return None
        v = self.feature_res_spinbox.value() / 100.0
        return v if v < e.effective_pixel_size / 1.05 else None

    # --- Map info line, cursor coordinates, shared zoom/pan ---
    def _map_viewers(self):
        return [self.viewer, self.discover_viewer, self.class_viewer, self.acc_viewer]

    def _map_header(self, rows):
        """ The control strip above a map: rows (layouts), centred in a strip as tall as the
        tallest tab's (set in init_main_layout). """
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addStretch(1)
        for r in rows:
            lay.addLayout(r)
        lay.addStretch(1)
        self._map_headers.append(w)
        return w

    def _map_tip(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #9a9a9a; font-size: 11px;")
        lbl.setToolTip(text)
        # Never widens the window: clipped when the map is narrow (full text in the tooltip).
        lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        lbl.setMinimumWidth(1)
        self._map_tips.append(lbl)
        return lbl

    def _new_map_info(self):
        # Takes the space beside the map controls, never widens the window, and shortens
        # with "..." when narrow; the full details are in its tooltip.
        lbl = ElidedInfoLabel()
        self._map_infos.append(lbl)
        return lbl

    def _crs_label(self, crs):
        if crs is None:
            return "no CRS"
        key = crs.to_wkt()
        if key not in self._crs_name_cache:
            name = ""
            try:
                from pyproj import CRS as _PCRS
                name = _PCRS.from_wkt(key).name
            except Exception:
                pass
            epsg = crs.to_epsg()
            label = " ".join(x for x in (f"EPSG:{epsg}" if epsg else "", name) if x)
            self._crs_name_cache[key] = label or "custom CRS"
        return self._crs_name_cache[key]

    def _ground_size(self, v):
        """ A pixel size in the CRS's units, readable: cm or m for metres, else degrees. """
        if v is None:
            return "?"
        if self.engine.native_crs_is_geographic:
            return f"{v:.2e}°"
        return f"{v * 100:.2f} cm" if v < 1 else f"{v:.2f} m"

    def _update_map_info(self):
        """ Line 1 of every map's info label: native and working resolution, size, CRS,
        bands and bit depth, DSM. Rebuilt only when the image, grid or DSM changes. """
        e = self.engine
        grid = None if e.ortho_data is None else tuple(e.ortho_data.shape[1:])
        key = (e.ortho_path, e.effective_pixel_size, grid, getattr(e, "dsm_path", None), e.has_dsm)
        if key == self._map_info_key:
            return
        self._map_info_key = key
        if grid is None or e.ortho_meta is None:
            self._map_info_line = ""
        else:
            h, w = grid
            crs = e.ortho_meta.get("crs")
            epsg = crs.to_epsg() if crs is not None else None
            parts = [f"Native {self._ground_size(e.native_pixel_size)}",
                     f"working {self._ground_size(e.effective_pixel_size)}",
                     f"{w:,} × {h:,} px",
                     f"EPSG:{epsg}" if epsg else "custom CRS"]
            details = [f"Orthomosaic: {os.path.basename(e.ortho_path or '')}",
                       f"Native pixel size {self._ground_size(e.native_pixel_size)}, working "
                       f"{self._ground_size(e.effective_pixel_size)} ({w:,} × {h:,} px)",
                       f"CRS: {self._crs_label(crs)}"]
            dt = getattr(e, "source_dtype", None)
            if dt:
                bits = {"uint8": "8-bit", "uint16": "16-bit", "int16": "16-bit",
                        "float32": "32-bit float", "float64": "64-bit float"}.get(dt, dt)
                parts.append(bits if dt == "uint8" else f"{bits} → 8-bit")
                details.append(f"{getattr(e, 'source_bands', 3)} bands, {bits}"
                               + ("" if dt == "uint8" else "; shown and analysed as 8-bit"))
            if e.has_dsm and getattr(e, "dsm_path", None):
                try:
                    with rasterio.open(e.dsm_path) as src:
                        gsd = self._ground_size(abs(src.transform.a))
                    parts.append(f"DSM {gsd}")
                    details.append(f"DSM: {os.path.basename(e.dsm_path)}, {gsd}")
                except Exception:
                    parts.append("DSM")
            self._map_info_line = "  ·  ".join(parts)
            self._map_info_details = "\n".join(details)
        self._set_map_info_text(self._cursor_text)

    def _set_map_info_text(self, cursor_line):
        # Two lines always, so the bar keeps its height while the cursor comes and goes.
        text = self._map_info_line + "\n" + (cursor_line or " ") if self._map_info_line else ""
        for lbl in self._map_infos:
            lbl.setText(text)
            lbl.setToolTip(getattr(self, "_map_info_details", "") if self._map_info_line else "")

    def _on_map_cursor(self, pt):
        e = self.engine
        if e.ortho_meta is None or e.ortho_data is None:
            return
        h, w = e.ortho_data.shape[1:]
        x, y = pt.x(), pt.y()
        if not (0 <= x < w and 0 <= y < h):
            self._cursor_text = ""
        else:
            X, Y = e.ortho_meta["transform"] * (x, y)
            if e.native_crs_is_geographic:
                where = f"Lon {X:.7f}°   Lat {Y:.7f}°"
            else:
                where = f"E {X:,.2f}   N {Y:,.2f}"
            self._cursor_text = f"{where}  ·  col {int(x):,}, row {int(y):,}"
        self._set_map_info_text(self._cursor_text)

    def _on_map_cursor_left(self):
        self._cursor_text = ""
        self._set_map_info_text("")

    def _on_view_changed(self, viewer):
        """ The visible map's zoom/pan becomes the shared view; maps on hidden tabs (redrawn
        or refitted in the background) don't overwrite it. """
        if viewer.isVisible():
            state = viewer.view_state()
            if state is not None:
                self._shared_view = state

    def _apply_shared_view(self):
        if self._shared_view is None:
            return
        for v in self._map_viewers():
            if v.isVisible():
                v.set_view_state(self._shared_view)

    # --- Segment size (m²) <-> segment count, and MMU px <-> m² ---
    DEFAULT_SEGMENT_SIZE_M2 = 10.0

    def _valid_area_m2(self):
        """ Valid image area on the working grid in m², or None (no image, geographic CRS). """
        e = self.engine
        if e.valid_mask is None or e.native_crs_is_geographic or not e.effective_pixel_size:
            return None
        return float(np.count_nonzero(e.valid_mask)) * e.effective_pixel_size ** 2

    def _set_segment_count_quietly(self, n):
        n = int(max(self.n_segments_spinbox.minimum(), min(self.n_segments_spinbox.maximum(), round(n))))
        self._seg_syncing = True
        try:
            self.n_segments_spinbox.setValue(n)
        finally:
            self._seg_syncing = False

    def _set_segment_size_quietly(self, m2):
        self.seg_size_spinbox.blockSignals(True)
        self.seg_size_spinbox.setValue(m2)
        self.seg_size_spinbox.blockSignals(False)

    def _segment_size_edited(self, m2):
        self._seg_size_drives = True
        area = self._valid_area_m2()
        if area and m2 > 0:
            self._set_segment_count_quietly(area / m2)

    def _segment_count_edited(self, n):
        if self._seg_syncing:
            return
        self._seg_size_drives = False
        area = self._valid_area_m2()
        if area and n > 0:
            self._set_segment_size_quietly(area / n)

    def _sync_segment_size(self):
        """ After a new image or grid: the size box is shown for projected CRSs only, and
        whichever of size / count the user set last is kept, the other recomputed. """
        area = self._valid_area_m2()
        self.seg_size_label.setVisible(area is not None)
        self.seg_size_spinbox.setVisible(area is not None)
        if area is None:
            return
        if self._seg_size_drives:
            self._set_segment_count_quietly(area / self.seg_size_spinbox.value())
        else:
            self._set_segment_size_quietly(area / max(1, self.n_segments_spinbox.value()))

    def _pixel_area_m2(self):
        e = self.engine
        if e.native_crs_is_geographic or not e.effective_pixel_size:
            return None
        return e.effective_pixel_size ** 2

    def _mmu_px_changed(self, px):
        a = self._pixel_area_m2()
        if a:
            self.gen_m2_spinbox.blockSignals(True)
            self.gen_m2_spinbox.setValue(px * a)
            self.gen_m2_spinbox.blockSignals(False)

    def _mmu_m2_changed(self, m2):
        a = self._pixel_area_m2()
        if a:
            self.gen_spinbox.setValue(int(round(m2 / a)))

    def _sync_mmu_m2(self):
        """ The MMU's m² box follows the pixel size of the working grid. """
        a = self._pixel_area_m2()
        self.gen_m2_spinbox.setVisible(a is not None)
        self.gen_m2_label.setVisible(a is not None)
        if a is None:
            return
        self.gen_m2_spinbox.blockSignals(True)
        self.gen_m2_spinbox.setRange(0.0, self.gen_spinbox.maximum() * a)
        self.gen_m2_spinbox.setSingleStep(max(0.1, round(100 * a, 1)))
        self.gen_m2_spinbox.setValue(self.gen_spinbox.value() * a)
        self.gen_m2_spinbox.blockSignals(False)

    def update_segmentation_plan_label(self, *_):
        """ Under the segment count: segment size on the ground, and whether segmentation
        runs in one go or in tiles (and roughly what it needs). """
        e = self.engine
        if e.ortho_data is None or e.valid_mask is None:
            self.seg_plan_label.setText("")
            return
        n = self.n_segments_spinbox.value()
        plan = e.plan_segmentation(n, self.gpu_radio.isChecked() and cuda_available())
        px = e.effective_pixel_size or 1.0
        if e.native_crs_is_geographic:
            size = f"~{plan['size_px']:.0f} px per segment"
        else:
            size = f"~{plan['size_px'] * px * px:.1f} m² ({plan['size_px']:.0f} px) per segment"
        how = "one run" if plan["tile"] is None else f"{plan['n_tiles']} tiles of {plan['tile']} px"
        fp = self._chosen_feature_pixel_size()
        feat = f"; colour/texture at {fp * 100:.2f} cm/px" if fp else ""
        self.seg_plan_label.setText(f"{size}, {how}{feat}.")

    def run_segmentation(self):
        if self.engine.ortho_data is None:
            print("Please load an image first.")
            return

        # What stays in memory after segmenting (segment map, DSM copies, adjacency...).
        h, w = self.engine.ortho_data.shape[1:]
        need = working_bytes(h, w, self.engine.has_dsm) - 4 * h * w   # image + mask already held
        ram = free_ram_bytes()
        if ram is not None and need > 0.8 * ram:
            reply = QMessageBox.warning(
                self, "Not Enough Memory",
                f"Segmenting this {w:,} x {h:,} px image needs about {need / 1e9:.1f} GB more memory, "
                f"but {ram / 1e9:.1f} GB is free. Resample to a coarser resolution first (Tab 1), "
                f"or close other programs.\n\nSegment anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        self.engine.feature_pixel_size = self._chosen_feature_pixel_size()

        force_device = "gpu" if self.gpu_radio.isChecked() else "cpu"
        self.run_seg_btn.setEnabled(False)
        self.run_seg_btn.setText("Processing...")
        self.engine.cancel_requested = False
        self.cancel_seg_btn.setEnabled(True)
        self.cancel_seg_btn.setText("Cancel")
        self.cancel_seg_btn.setVisible(True)

        self.seg_thread = QThread()
        self.seg_worker = SegmentationWorker(
            self.engine,
            self.n_segments_spinbox.value(),
            self.compactness_dspinbox.value(),
            self.sigma_dspinbox.value(),
            self,
            force_device=force_device
        )
        self.seg_worker.moveToThread(self.seg_thread)

        self.seg_thread.started.connect(self.seg_worker.run)
        self.seg_worker.finished.connect(self.on_segmentation_finished)
        self.seg_worker.failed.connect(self.on_segmentation_failed)
        self.seg_worker.cancelled.connect(self.on_segmentation_cancelled)

        self.seg_worker.finished.connect(self.seg_thread.quit)
        self.seg_worker.failed.connect(self.seg_thread.quit)
        self.seg_worker.cancelled.connect(self.seg_thread.quit)
        self.seg_worker.finished.connect(self.seg_worker.deleteLater)
        self.seg_thread.finished.connect(self.seg_thread.deleteLater)

        self.seg_thread.start()

    def rebuild_sample_class_map(self):
        """ Training samples changed structurally (class added/removed/recoloured, many
        samples at once): rebuild the samples layer on the next display update. Single
        clicks use _update_sample_segment instead. """
        self.sample_class_map = None
        self._samples_item = None

    def generate_boundary_and_visual_caches(self):
        """ Rebuilds base_seg_qimg / boundary_overlay_qimg from engine.segmented_image (e.g. after restoring a saved workspace session, where the background worker never ran). """
        if self.engine.segmented_image is None or self.engine.original_ortho_data is None:
            return
        self.base_seg_qimg, self.boundary_overlay_qimg = build_boundary_and_visual_caches(self.engine)
        self._invalidate_display_caches()

    def cancel_segmentation(self):
        self.engine.cancel_requested = True
        self.cancel_seg_btn.setEnabled(False)
        self.cancel_seg_btn.setText("Cancelling…")
        print("Cancelling segmentation at the next checkpoint...")

    def _seg_controls_idle(self):
        self.run_seg_btn.setEnabled(True)
        self.run_seg_btn.setText("Run Segmentation")
        self.cancel_seg_btn.setVisible(False)
        self.engine.cancel_requested = False

    def on_segmentation_cancelled(self):
        self._seg_controls_idle()
        print(">>> Segmentation cancelled; the previous result (if any) is unchanged. <<<")

    def on_segmentation_finished(self, segmented, base_seg_qimg, boundary_overlay_qimg, num_polygons, duration):
        self._seg_controls_idle()

        self.base_seg_qimg = base_seg_qimg
        self.boundary_overlay_qimg = boundary_overlay_qimg

        # A fresh segmentation means every previous segment ID -- and anything trained
        # or classified against those old IDs -- no longer applies.
        self.classified_image_data = None
        self.raw_classified_raster_ints = None
        self.sieved_classified_raster_ints = None
        self.classifier = None

        self._discover_reset()  # groups and deep features belong to the old segments
        if self.iterations:
            print("Notice: the Iterations list was cleared -- its snapshots belong to the previous segmentation.")
        self.iterations = []
        self.refresh_iteration_table()

        self._invalidate_display_caches()

        # A fresh segmentation invalidates every previous segment ID.
        self.selected_segments = set()
        self.locked_selection = set()
        self.current_seed_segments = set()
        self.selection_undo_stack = []
        self.selection_count_label.setText("Selected: 0 segments")

        self.show_boundaries_check.setEnabled(True)
        self.refresh_load_view_overlay()

        print("[Step 5/5] Opening Discover Classes...")
        QApplication.processEvents()
        self.setup_classification_view()
        self._release_after_segmentation()
        self.tabs.setCurrentWidget(self.discover_tab)

        print(f"\n>>> PROCESSING COMPLETE! Compiled {num_polygons} polygons in {duration:.1f} seconds. <<<")

    def _release_after_segmentation(self):
        """ Frees what segmentation needed but the next steps don't: the DSM/slope copies on
        the working grid (features are computed; rebuilt automatically if a segment merge
        needs them), GPU memory left by a GPU attempt, and collectable garbage. """
        freed = self.engine.dsm_grid_bytes() if self.engine.__dict__.get("_dsm_grid") is not None else 0
        self.engine.dsm_data = None
        try:
            if od.torch is not None and od.torch.cuda.is_available():
                od.torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        if freed:
            print(f" -> Freed {freed / 1e9:.1f} GB (DSM grid copies; rebuilt automatically if needed).")

    def on_segmentation_failed(self, traceback_str):
        self._seg_controls_idle()
        print(f"Segmentation Thread Error:\n{traceback_str}")

    # --- TAB 2: Discover Classes ---
    def init_discover_tab(self):
        """ Cluster-then-label: group segments that look alike (deep appearance features
        and/or the classifier's features), let the user name, merge, split or ignore the
        groups, then send named groups to Classification as training samples or map them
        directly. Engine-side logic lives in obia_discover.py. """
        tab = QWidget()
        self.discover_tab = tab
        layout = QHBoxLayout(tab)

        col1 = QWidget()
        col1_layout = QVBoxLayout(col1)
        col1.setFixedWidth(self.LEFT_COL1_W)

        find_group = QGroupBox("Find Groups")
        fg = QVBoxLayout()
        intro = QLabel("Groups segments that look alike. Name the groups that mean something; "
                       "merge, split or ignore the rest.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #9a9a9a;")
        fg.addWidget(intro)
        self.disc_use_deep_check = QCheckBox("Appearance (deep features, DINOv2)")
        self.disc_use_deep_check.setChecked(od.torch is not None)
        self.disc_use_deep_check.setEnabled(od.torch is not None)
        self.disc_use_feat_check = QCheckBox("Colour, texture, shape && height (classifier features)")
        self.disc_use_feat_check.setChecked(True)
        fg.addWidget(self.disc_use_deep_check)
        fg.addWidget(self.disc_use_feat_check)
        n_row = QHBoxLayout()
        n_row.addWidget(QLabel("Number of groups:"))
        self.disc_n_slider = QSlider(Qt.Horizontal)
        self.disc_n_slider.setRange(5, 60)
        self.disc_n_slider.setValue(30)
        self.disc_n_spin = QSpinBox()
        self.disc_n_spin.setRange(5, 60)
        self.disc_n_spin.setValue(30)
        self.disc_n_slider.valueChanged.connect(self.disc_n_spin.setValue)
        self.disc_n_spin.valueChanged.connect(self.disc_n_slider.setValue)
        n_row.addWidget(self.disc_n_slider)
        n_row.addWidget(self.disc_n_spin)
        fg.addLayout(n_row)
        self.disc_edge_check = QCheckBox("Leave out segments at the image edge")
        self.disc_edge_check.setChecked(True)
        fg.addWidget(self.disc_edge_check)
        self.disc_find_btn = QPushButton("Find Groups")
        self.disc_find_btn.setMinimumHeight(38)
        self.disc_find_btn.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: white; font-weight: 600; border-radius: 4px; }"
            "QPushButton:hover { background-color: #a93226; }"
            "QPushButton:disabled { background-color: #d3a29c; color: #f2e6e4; }")
        self.disc_find_btn.clicked.connect(self.discover_find_groups)
        fg.addWidget(self.disc_find_btn)
        self.disc_cancel_btn = QPushButton("Cancel")
        self.disc_cancel_btn.setToolTip("Stop computing deep features")
        self.disc_cancel_btn.clicked.connect(self.cancel_deep_features)
        self.disc_cancel_btn.setVisible(False)
        fg.addWidget(self.disc_cancel_btn)
        self.disc_status_label = QLabel("Segment an image first (Tab 1).")
        self.disc_status_label.setWordWrap(True)
        self.disc_status_label.setStyleSheet("color: #8fbf9a;")
        fg.addWidget(self.disc_status_label)
        find_group.setLayout(fg)

        groups_box = QGroupBox("Groups")
        self.disc_groups_box = groups_box
        gl = QVBoxLayout()
        filt_row = QHBoxLayout()
        self.disc_filter_combo = QComboBox()
        self.disc_filter_combo.addItems(["All groups", "Unnamed", "Named", "Ignored"])
        self.disc_filter_combo.currentIndexChanged.connect(lambda _: self.refresh_discover_list())
        self.disc_sort_combo = QComboBox()
        self.disc_sort_combo.addItems(["Largest first", "Look-alikes together"])
        self.disc_sort_combo.currentIndexChanged.connect(lambda _: self.refresh_discover_list())
        filt_row.addWidget(self.disc_filter_combo)
        filt_row.addWidget(self.disc_sort_combo)
        gl.addLayout(filt_row)
        self.disc_cards_host = QWidget()
        self.disc_cards_layout = QVBoxLayout(self.disc_cards_host)
        self.disc_cards_layout.setContentsMargins(0, 0, 0, 0)
        self.disc_cards_layout.setSpacing(4)
        self.disc_cards_layout.addStretch(1)
        cards_scroll = QScrollArea()
        cards_scroll.setWidgetResizable(True)
        cards_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        cards_scroll.setWidget(self.disc_cards_host)
        gl.addWidget(cards_scroll, 1)
        hint = QLabel("Ctrl+click cards to select several, then Merge.")
        hint.setStyleSheet("color: #9a9a9a; font-size: 11px;")
        gl.addWidget(hint)
        groups_box.setLayout(gl)

        self.disc_send_btn = QPushButton("Send Named Groups to Classification")
        self.disc_send_btn.setMinimumHeight(36)
        self.disc_send_btn.setStyleSheet(
            "QPushButton { background-color: #2f6fbf; color: white; font-weight: 600; border-radius: 4px; }"
            "QPushButton:hover { background-color: #285fa3; }"
            "QPushButton:disabled { background-color: #8a9bb0; color: #e3e8ee; }")
        self.disc_send_btn.clicked.connect(self.send_groups_to_classification)

        col1_layout.addWidget(find_group)
        col1_layout.addWidget(groups_box, 1)
        col1_layout.addWidget(self.disc_send_btn)

        col2 = QWidget()
        col2_layout = QVBoxLayout(col2)
        col2.setFixedWidth(self.LEFT_COL2_W)
        sel_box = QGroupBox("Selected Group")
        sl = QVBoxLayout()
        self.disc_sel_header = QLabel("No group selected.")
        self.disc_sel_header.setStyleSheet("font-size: 15px; font-weight: 600;")
        sl.addWidget(self.disc_sel_header)
        self.disc_gallery = ChipGallery(chip_size=86)
        self.disc_gallery.contextRequested.connect(self.on_discover_gallery_context)
        self.disc_gallery.itemDoubleClicked.connect(self.on_discover_tile_double_clicked)
        sl.addWidget(self.disc_gallery, 1)
        self.disc_grid_caption = QLabel("")
        self.disc_grid_caption.setWordWrap(True)
        self.disc_grid_caption.setStyleSheet("color: #9a9a9a; font-size: 11px;")
        sl.addWidget(self.disc_grid_caption)
        sl.addWidget(QLabel("Name this group:"))
        name_row = QHBoxLayout()
        self.disc_name_combo = ClassNameCombo()
        self.disc_name_combo.classChosen.connect(self.discover_assign_class)
        self.disc_name_combo.specialChosen.connect(self.discover_assign_special)
        self.disc_name_combo.removeRequested.connect(self.discover_remove_class)
        self.disc_name_combo.renameRequested.connect(self.discover_rename_class)
        self.disc_name_combo.recolorRequested.connect(self.discover_recolor_class)
        self.disc_name_combo.lineEdit().textEdited.connect(self._discover_name_typed)
        self.disc_name_combo.lineEdit().returnPressed.connect(self.discover_add_typed_class)
        self.disc_add_class_btn = QPushButton("Add Class")
        self.disc_add_class_btn.setStyleSheet(
            "QPushButton { background-color: #2f6fbf; color: white; font-weight: 600; padding: 3px 10px; border-radius: 3px; }")
        self.disc_add_class_btn.clicked.connect(self.discover_add_typed_class)
        self.disc_add_class_btn.hide()
        name_row.addWidget(self.disc_name_combo, 1)
        name_row.addWidget(self.disc_add_class_btn)
        sl.addLayout(name_row)
        self.disc_name_feedback = QLabel("")
        self.disc_name_feedback.setWordWrap(True)
        self.disc_name_feedback.setStyleSheet("color: #8fbf9a; font-size: 11px;")
        sl.addWidget(self.disc_name_feedback)
        btn_grid = QGridLayout()
        self.disc_merge_btn = QPushButton("Merge…")
        self.disc_merge_btn.clicked.connect(self.discover_merge)
        self.disc_split_btn = QPushButton("Split…")
        self.disc_split_btn.clicked.connect(self.discover_split)
        self.disc_ignore_btn = QPushButton("Ignore Group")
        self.disc_ignore_btn.clicked.connect(self.discover_toggle_ignore)
        self.disc_similar_btn = QPushButton("Find Similar Elsewhere")
        self.disc_similar_btn.clicked.connect(self.discover_find_similar_elsewhere)
        btn_grid.addWidget(self.disc_merge_btn, 0, 0)
        btn_grid.addWidget(self.disc_split_btn, 0, 1)
        btn_grid.addWidget(self.disc_ignore_btn, 1, 0)
        btn_grid.addWidget(self.disc_similar_btn, 1, 1)
        sl.addLayout(btn_grid)
        self.disc_close_label = QLabel("")
        self.disc_close_label.setWordWrap(True)
        self.disc_close_label.setStyleSheet("background:#2f2a22; border:1px solid #6b5a33; padding:5px; color:#e8d6a8;")
        self.disc_close_label.hide()
        sl.addWidget(self.disc_close_label)
        sel_box.setLayout(sl)
        col2_layout.addWidget(sel_box, 1)

        viewer_container = QWidget()
        vl = QVBoxLayout(viewer_container)
        vl.setContentsMargins(0, 0, 0, 0)
        mode_row = QHBoxLayout()
        self.disc_map_label = QLabel("View:")
        mode_row.addWidget(self.disc_map_label)
        self.disc_mode_selected = QRadioButton("Selected group")
        self.disc_mode_all = QRadioButton("All groups")
        self.disc_mode_named = QRadioButton("Named only")
        self.disc_mode_selected.setChecked(True)
        self.disc_mode_group = QButtonGroup(self)
        for rb in (self.disc_mode_selected, self.disc_mode_all, self.disc_mode_named):
            self.disc_mode_group.addButton(rb)
            mode_row.addWidget(rb)
            rb.toggled.connect(lambda on: on and self.update_discover_map())
        mode_row.addWidget(self._new_map_info(), 1)
        ctl_row = QHBoxLayout()
        self.disc_overlay_check = QCheckBox("Group overlay")
        self.disc_overlay_check.setChecked(True)
        self.disc_overlay_check.toggled.connect(lambda _: self._apply_discover_layer_visibility())
        ctl_row.addWidget(self.disc_overlay_check)
        ctl_row.addWidget(QLabel("Opacity:"))
        self.disc_opacity_slider = QSlider(Qt.Horizontal)
        self.disc_opacity_slider.setRange(0, 100)
        self.disc_opacity_slider.setValue(75)
        self.disc_opacity_slider.setMaximumWidth(140)
        self.disc_opacity_slider.setTracking(False)   # applies on release
        self.disc_opacity_slider.valueChanged.connect(lambda _: self._apply_discover_layer_visibility())
        ctl_row.addWidget(self.disc_opacity_slider)
        self.disc_edges_check = QCheckBox("Segment edges")
        self.disc_edges_check.toggled.connect(lambda _: self._apply_discover_layer_visibility())
        ctl_row.addWidget(self.disc_edges_check)
        ctl_row.addStretch(1)
        vl.addWidget(self._map_header([mode_row, ctl_row]))
        self.discover_viewer = InteractivePhotoViewer()
        self.discover_viewer.set_detail_provider(self._detail_image)
        self.discover_viewer.photoClicked.connect(self.on_discover_map_clicked)
        self.discover_viewer.contextMenuRequested.connect(self.on_discover_map_context)
        vl.addWidget(self.discover_viewer, 1)
        vl.addWidget(self._map_tip("Click a segment to select its group; right-click to move it to another group. "
                                   "Double-click a tile to zoom to it."))

        layout.addWidget(col1)
        layout.addWidget(col2)
        layout.addWidget(viewer_container, 1)
        self.tabs.addTab(tab, "2. Discover Classes")
        self._discover_set_enabled(False)

    # --- Discover: state helpers ---
    def _discover_reset(self):
        """ Forget groups and cached deep features (a new segmentation or session). """
        self.deep = None
        self.groups = None
        self.disc_selected = None
        self.disc_multi = []
        self.disc_chip_offset = 0
        self._seg_index = None
        self._sim_space = None
        self._strip_cache = {}
        self._disc_cards = {}
        if hasattr(self, "disc_cards_layout"):
            self.refresh_discover_list()
            self.show_discover_group(None)
            self.discover_viewer.set_image(self.raw_pixmap if self.raw_pixmap is not None else QImage(), grid_size=self.raw_grid)
            seg_ok = self.engine.segmented_image is not None
            self.disc_status_label.setText("Ready: press Find Groups." if seg_ok else "Segment an image first (Tab 1).")
            self._discover_set_enabled(False)

    def _discover_set_enabled(self, has_groups):
        for w in (self.disc_send_btn, self.disc_merge_btn, self.disc_split_btn,
                  self.disc_ignore_btn, self.disc_similar_btn, self.disc_name_combo, self.disc_gallery):
            w.setEnabled(has_groups)

    def _segment_index(self):
        """ (ids, centroids, id -> row, fingerprint) for the current segmentation, cached. """
        if self._seg_index is None and self.engine.segmented_image is not None:
            seg = self.engine.segmented_image
            ids, cents = od.segment_centroids(seg)
            self._seg_index = (ids, cents, {int(s): i for i, s in enumerate(ids)}, od.seg_fingerprint(seg))
        return self._seg_index

    def _segment_at(self, pos):
        seg = self.engine.segmented_image
        if seg is None:
            return 0
        x, y = int(pos.x()), int(pos.y())
        if 0 <= y < seg.shape[0] and 0 <= x < seg.shape[1]:
            return int(seg[y, x])
        return 0

    def _chip(self, seg_id, size):
        """ A tile of one segment and its surroundings: a square window 3x the segment's
        larger side (so it fills the middle third), at least 32 px across -- the segment
        reads the same at any pixel size, unlike a fixed window on the ground. """
        ids, cents, pos, _ = self._segment_index()
        boxes = self._seg_slices()
        sl = boxes[int(seg_id) - 1] if int(seg_id) - 1 < len(boxes) else None
        side = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) if sl is not None else 20
        half = max(16, int(round(1.5 * side)))
        img = make_chip_qimage(self.engine.original_ortho_data, self.engine.segmented_image,
                               int(seg_id), cents[pos[int(seg_id)]], half, size)
        self._draw_tile_scale(img, 2 * half, size)
        return img

    def _draw_tile_scale(self, img, window_px, size):
        """ A scale bar in the tile's bottom-left corner: tiles are zoomed to their segment,
        so without it a large and a small segment look the same. Labelled from 90 px up. """
        e = self.engine
        if e.native_crs_is_geographic or not e.effective_pixel_size:
            return
        m_per_px = window_px * e.effective_pixel_size / size          # metres per tile pixel
        target = 0.28 * size * m_per_px
        nice = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
        length = min(nice, key=lambda v: abs(v - target))
        bar = max(4, int(round(length / m_per_px)))
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing, False)
        x, y = 4, size - 5
        p.fillRect(x - 1, y - 2, bar + 2, 5, QColor(0, 0, 0, 170))
        p.fillRect(x, y - 1, bar, 3, QColor(255, 255, 255))
        if size >= 90:
            f = p.font()
            f.setPixelSize(10)
            f.setBold(True)
            p.setFont(f)
            label = f"{length:g} m"
            p.setPen(QColor(0, 0, 0, 200))
            p.drawText(x + 1, y - 5, label)
            p.setPen(QColor(255, 255, 255))
            p.drawText(x, y - 6, label)
        p.end()

    def _tile_tip(self, seg_id):
        fd = self.engine.features_df
        if fd is not None and seg_id in fd.index and "area" in fd.columns:
            return f"Segment #{seg_id} · {fd.at[seg_id, 'area']:,.1f} m²"
        return f"Segment #{seg_id}"

    def _deep_for_current(self):
        """ Cached deep features restricted to the current segments, or None. """
        if self.deep is None or self.engine.segmented_image is None:
            return None
        ids, _, _, fp = self._segment_index()
        if self.deep.get("fingerprint") != fp:
            return None
        return od.subset_deep(self.deep, ids)

    def _discover_after_segment_change(self):
        """ After a manual segment merge: drop vanished segments from the groups. """
        self._seg_index = None
        self._sim_space = None
        ids, _, _, fp = self._segment_index()
        if self.deep is not None:
            sub = od.subset_deep(self.deep, ids)
            self.deep = None if sub is None else dict(sub, fingerprint=fp)
        if self.groups is not None:
            self.groups.settings["fingerprint"] = fp
            dropped = self.groups.keep_ids(ids)
            if dropped:
                self._strip_cache = {}
                self.refresh_discover_list()

    # --- Discover: finding groups ---
    def discover_find_groups(self):
        if self.engine.segmented_image is None or self.engine.features_df is None:
            print("Discover: segment the image first (Tab 1).")
            return
        if not (self.disc_use_deep_check.isChecked() or self.disc_use_feat_check.isChecked()):
            print("Discover: tick at least one kind of feature to group on.")
            return
        if self.groups is not None and any(g["status"] != "unnamed" for g in self.groups.groups.values()):
            reply = QMessageBox.question(
                self, "Find Groups",
                "Finding groups again replaces the current groups, including their names, merges and splits. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        if self.disc_use_deep_check.isChecked() and self._deep_for_current() is None:
            if not self._confirm_cpu_deep_time():
                return
            if self.disc_use_deep_check.isChecked():
                self._start_deep_features()
                return
        self._discover_group_now()

    # Seconds per segment for DINOv2 (two patches each), measured on the v1.0 builds:
    # CPU ~0.11 s (8-core desktop), CUDA ~0.008 s.
    DEEP_S_PER_SEGMENT_CPU = 0.11

    def _confirm_cpu_deep_time(self):
        """ Without a CUDA GPU, appearance features can take a long time on a large scene.
        Above ~2 minutes, say how long and offer to group on the standard features instead.
        Returns False to cancel (Appearance may have been unticked to go on without it). """
        if cuda_available():
            return True
        n = len(self._segment_index()[0])
        est = n * self.DEEP_S_PER_SEGMENT_CPU
        if est < 120:
            return True
        box = QMessageBox(self)
        box.setWindowTitle("Find Groups on the CPU")
        box.setIcon(QMessageBox.Information)
        box.setText(f"Computing the appearance features (DINOv2) for {n:,} segments on the CPU will take "
                    f"about {self._fmt_duration(est)}.")
        box.setInformativeText("It is done once per segmentation and saved with the workspace. With an NVIDIA "
                               "GPU (GPU edition) it takes a few minutes. You can also group on colour, texture, "
                               "shape and height only, and add appearance later.")
        go = box.addButton("Compute appearance", QMessageBox.AcceptRole)
        without = box.addButton("Group without appearance", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(go)
        box.exec()
        if box.clickedButton() is go:
            return True
        if box.clickedButton() is without:
            if not self.disc_use_feat_check.isChecked():
                self.disc_use_feat_check.setChecked(True)
            self.disc_use_deep_check.setChecked(False)
            return True
        return False

    @staticmethod
    def _fmt_duration(s):
        s = int(round(s))
        if s < 90:
            return f"{s} s"
        m = int(round(s / 60))
        return f"{m} min" if m < 90 else f"{m // 60} h {m % 60:02d} min"

    def _start_deep_features(self):
        self.disc_find_btn.setEnabled(False)
        self.disc_find_btn.setText("Computing deep features…")
        n = len(self._segment_index()[0])
        self._deep_t0 = time.time()
        print(f"Computing deep appearance features for {n} segments (DINOv2, {', '.join(f'{int(m)} m' for m in od.PATCH_SIZES_M)} patches)...")
        self.deep_thread = QThread()
        self.deep_worker = DeepFeatureWorker(self.engine)
        self.deep_worker.moveToThread(self.deep_thread)
        self.deep_thread.started.connect(self.deep_worker.run)
        self.deep_worker.progress.connect(self._on_deep_progress)
        self.deep_worker.finished.connect(self._on_deep_finished)
        self.deep_worker.failed.connect(self._on_deep_failed)
        self.deep_worker.cancelled.connect(self._on_deep_cancelled)
        self.deep_worker.finished.connect(self.deep_thread.quit)
        self.deep_worker.failed.connect(self.deep_thread.quit)
        self.deep_worker.cancelled.connect(self.deep_thread.quit)
        self.disc_cancel_btn.setEnabled(True)
        self.disc_cancel_btn.setText("Cancel")
        self.disc_cancel_btn.setVisible(True)
        self.deep_worker.finished.connect(self.deep_worker.deleteLater)
        self.deep_thread.finished.connect(self.deep_thread.deleteLater)
        self.deep_thread.start()

    def _on_deep_progress(self, done, total):
        eta = ""
        t0 = getattr(self, "_deep_t0", None)
        if t0 and done > 0 and done < total:
            left = (time.time() - t0) * (total - done) / done
            eta = f" · about {self._fmt_duration(left)} left"
        self.disc_status_label.setText(f"Deep features: {done:,} / {total:,} patches{eta}")

    def cancel_deep_features(self):
        if getattr(self, "deep_worker", None) is not None:
            self.deep_worker.cancel_requested = True
            self.disc_cancel_btn.setEnabled(False)
            self.disc_cancel_btn.setText("Cancelling…")

    def _deep_controls_idle(self):
        self.disc_find_btn.setEnabled(True)
        self.disc_find_btn.setText("Find Groups")
        self.disc_cancel_btn.setVisible(False)

    def _on_deep_cancelled(self):
        self._deep_controls_idle()
        self.deep_worker = None
        print("Deep features cancelled.")
        self.disc_status_label.setText("Cancelled. Press Find Groups to start again.")

    def _on_deep_finished(self, deep):
        self._deep_controls_idle()
        self.deep = deep
        print(f" -> Deep features done on {deep['device'].upper()} in {deep['seconds']:.0f} s.")
        if self.workspace_dir:
            try:
                self._ensure_workspace_dir()
                od.save_deep_features(os.path.join(self.workspace_dir, "deep_features.npz"), deep)
                print(f" -> Cached in {os.path.join(self.workspace_dir, 'deep_features.npz')}")
            except Exception as e:
                print(f" -> Could not cache deep features ({e}); they will be recomputed next session.")
        self._discover_group_now()

    def _on_deep_failed(self, tb):
        self._deep_controls_idle()
        print(f"Deep features failed:\n{tb}")
        self.disc_status_label.setText("Deep features failed (see log). Untick Appearance to group on the classifier features only.")

    def _discover_group_now(self):
        ids_all, cents, _, fp = self._segment_index()
        ids = np.array([s for s in ids_all if s in self.engine.features_df.index])
        deep = self._deep_for_current() if self.disc_use_deep_check.isChecked() else None
        use_deep = deep is not None
        use_feat = self.disc_use_feat_check.isChecked()
        if not use_deep and not use_feat:
            print("Discover: no features available to group on.")
            return
        if deep is not None:
            pos = {int(s): i for i, s in enumerate(deep["ids"])}
            edge = deep["edge"][[pos[int(s)] for s in ids]]
        else:
            pos = self._segment_index()[2]
            edge = od.edge_flags(cents[[pos[int(s)] for s in ids]], self.engine.valid_mask,
                                 self.engine.effective_pixel_size or 0.1)
        X = od.build_space(ids, self.engine.features_df, self.engine.active_feature_columns(include_deep=False), deep, use_deep, use_feat)
        n = self.disc_n_spin.value()
        exclude = edge if self.disc_edge_check.isChecked() else None
        settings = {"use_deep": use_deep, "use_features": use_feat, "n_groups": n,
                    "exclude_edge": bool(exclude is not None), "fingerprint": fp}
        print(f"Grouping {len(ids):,} segments into {n} groups...")
        QApplication.processEvents()
        self.groups = od.GroupModel.find(ids, X, n, exclude=exclude, settings=settings)
        self._sim_space = None
        self._strip_cache = {}
        self.disc_multi = []
        left_out = int((self.groups.labels == "").sum())
        basis = " + ".join(b for b, on in (("deep features", use_deep), ("classifier features", use_feat)) if on)
        self.disc_status_label.setText(f"{len(ids):,} segments · {len(self.groups.groups)} groups · {basis}"
                                       + (f" · {left_out} edge segments left out" if left_out else ""))
        sugg = self.groups.merge_suggestions(3)
        if sugg:
            print(" -> Look-alike groups (merge candidates): " + ", ".join(f"{a} + {b}" for a, b, _ in sugg))
        self._discover_set_enabled(True)
        self.refresh_discover_list()
        first = self._visible_keys()[0] if self._visible_keys() else None
        self.show_discover_group(first)
        self.discover_viewer.set_image(self.raw_pixmap if self.raw_pixmap is not None else QImage(), grid_size=self.raw_grid)
        self.update_discover_map()

    # --- Discover: list and selection ---
    def _visible_keys(self):
        if self.groups is None:
            return []
        g = self.groups
        keys = [k for k in g.keys() if g.size(k) > 0]
        flt = self.disc_filter_combo.currentText()
        if flt == "Unnamed":
            keys = [k for k in keys if g.groups[k]["status"] == "unnamed"]
        elif flt == "Named":
            keys = [k for k in keys if g.groups[k]["status"] == "named"]
        elif flt == "Ignored":
            keys = [k for k in keys if g.groups[k]["status"] == "ignored"]
        if self.disc_sort_combo.currentText() == "Look-alikes together" and len(keys) > 2:
            from scipy.cluster.hierarchy import linkage, leaves_list
            cents = np.array([g.centre(k) for k in keys])
            keys = [keys[i] for i in leaves_list(linkage(cents, "ward"))]
        else:
            keys.sort(key=lambda k: -g.size(k))
        return keys

    def _group_strip(self, key):
        if key not in self._strip_cache:
            chips = [self._chip(self.groups.ids[i], 46) for i in self.groups.typical(key, 5)]
            self._strip_cache[key] = chip_strip_pixmap(chips, 46)
        return self._strip_cache[key]

    def refresh_discover_list(self):
        """ Rebuilds the group cards (cheap: strips are cached per group). """
        while self.disc_cards_layout.count() > 1:
            item = self.disc_cards_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._disc_cards = {}
        if self.groups is None:
            self.disc_groups_box.setTitle("Groups")
            return
        g = self.groups
        sugg_pairs = {k for a, b, _ in g.merge_suggestions(4) for k in (a, b)}
        live = [k for k in g.keys() if g.size(k) > 0]
        self.disc_groups_box.setTitle(f"Groups ({len(live)})")
        for k in self._visible_keys():
            info = g.groups[k]
            card = GroupCard(k)
            n = g.size(k)
            if info["status"] == "named":
                tag, kind = "Named", "named"
            elif info["status"] == "ignored":
                tag, kind = "Ignored", "ignored"
            elif k in sugg_pairs:
                tag, kind = "Look-alike", "hint"
            else:
                tag, kind = "Unnamed", "unnamed"
            card.setToolTip(f"{k}: {n:,} segments")
            card.update_content(self._group_color(k), f"{n:,} segments", info["status"],
                                info["name"], tag, kind, self._group_strip(k))
            card.clicked.connect(self.on_discover_card_clicked)
            card.set_selected(k == self.disc_selected or k in self.disc_multi, multi=k in self.disc_multi and k != self.disc_selected)
            self.disc_cards_layout.insertWidget(self.disc_cards_layout.count() - 1, card)
            self._disc_cards[k] = card

    def on_discover_card_clicked(self, key, kind):
        order = self._visible_keys()
        anchor = getattr(self, "disc_anchor", None)
        if kind == "shift" and anchor in order and key in order:
            a, b = sorted((order.index(anchor), order.index(key)))
            self.disc_multi = order[a:b + 1]
        elif kind == "ctrl":
            if key in self.disc_multi:
                self.disc_multi.remove(key)
            else:
                if self.disc_selected and self.disc_selected not in self.disc_multi:
                    self.disc_multi.append(self.disc_selected)
                self.disc_multi.append(key)
            self.disc_anchor = key
        else:
            self.disc_multi = []
            self.disc_anchor = key
        self.show_discover_group(key)

    def _sync_card_selection(self):
        for k, card in self._disc_cards.items():
            card.set_selected(k == self.disc_selected or k in self.disc_multi,
                              multi=k in self.disc_multi and k != self.disc_selected)
        n = len(self.disc_multi)
        self.disc_merge_btn.setText(f"Merge Selected ({n})" if n >= 2 else "Merge…")

    def show_discover_group(self, key):
        self.disc_selected = key
        self.disc_chip_offset = 0
        if key is None or self.groups is None or key not in self.groups.groups:
            self.disc_sel_header.setText("No group selected.")
            self.disc_gallery.clear()
            self.disc_grid_caption.setText("")
            self.disc_close_label.hide()
            if hasattr(self, "_disc_cards"):
                self._sync_card_selection()
            return
        g = self.groups
        info = g.groups[key]
        self._update_sel_header()
        self._render_discover_grid()
        self._refresh_name_combo()
        self.disc_name_feedback.setText("")
        self.disc_ignore_btn.setText("Restore Group" if info["status"] == "ignored" else "Ignore Group")
        close = g.closest_groups(key, 2)
        if close:
            desc = [f"<b>{c}</b>" + (f" ({g.groups[c]['name']})" if g.groups[c]["name"] else "") for c in close]
            self.disc_close_label.setText("Looks closest to " + " and ".join(desc) + ". Compare them before naming.")
            self.disc_close_label.show()
        else:
            self.disc_close_label.hide()
        self._sync_card_selection()
        self.update_discover_map()

    def _render_discover_grid(self):
        g, key = self.groups, self.disc_selected
        idx = g.typical(key, g.size(key))
        self.disc_gallery.set_segments([int(g.ids[i]) for i in idx], self._chip, tip_fn=self._tile_tip)
        self.disc_grid_caption.setText(f"{len(idx):,} segments, most typical first. Scroll for more; "
                                       f"Ctrl/Shift+click tiles to select, right-click to move them.")

    def on_discover_gallery_context(self, global_pos):
        if self.groups is None or self.disc_selected is None:
            return
        sids = self.disc_gallery.selected_segment_ids()
        menu = QMenu(self)
        if not sids:
            head = menu.addAction("Select tiles first (click, Ctrl+click, Shift+click)")
            head.setEnabled(False)
            menu.addAction("Select All Tiles Loaded So Far", self.disc_gallery.selectAll)
            menu.exec(global_pos)
            return
        g = self.groups
        sub = menu.addMenu(f"Move {len(sids)} Tile(s) to Group")
        for k in self._visible_keys() if self.disc_filter_combo.currentText() == "All groups" else [k for k in g.keys() if g.size(k) > 0]:
            if k == self.disc_selected:
                continue
            name = g.groups[k]["name"]
            act = sub.addAction(f"{k}" + (f" · {name}" if name else "") + ("  (ignored)" if g.groups[k]["status"] == "ignored" else ""))
            act.triggered.connect(lambda _=False, k=k: self._move_tiles(sids, k))
        menu.addAction(f"Move {len(sids)} Tile(s) to a New Group", lambda: self._move_tiles(sids, g.new_group()))
        menu.addAction(f"Leave {len(sids)} Tile(s) Out of All Groups", lambda: self._move_tiles(sids, ""))
        menu.exec(global_pos)

    def _move_tiles(self, sids, key):
        g = self.groups
        idx = [g.pos[int(s)] for s in sids if int(s) in g.pos]
        for k in {g.labels[i] for i in idx} | {key, self.disc_selected}:
            self._strip_cache.pop(k, None)
        g.move(idx, key)
        print(f"Discover: moved {len(idx)} segment(s) to " + (key or "no group") + ".")
        self.refresh_discover_list()
        self.show_discover_group(self.disc_selected if self.disc_selected in g.groups else key)

    # --- Discover: edits ---
    def _update_sel_header(self):
        g, key = self.groups, self.disc_selected
        n = g.size(key)
        info = g.groups[key]
        label = info["name"] if info["status"] == "named" else ("ignored" if info["status"] == "ignored" else "unnamed")
        self.disc_sel_header.setText(
            f"<span style='color:{self._group_color(key)}'>■</span> {key} &nbsp;<span style='font-size:12px; "
            f"font-weight:400; color:#9a9a9a'>{n:,} segments · {label}</span>")

    def _group_color(self, key):
        """ A named group shows its class colour; otherwise its own group colour. """
        info = self.groups.groups[key]
        if info["status"] == "named" and info["name"] in self.classes:
            return self.classes[info["name"]]["color"].name()
        return info["color"]

    def _refresh_name_combo(self):
        info = self.groups.groups[self.disc_selected] if (self.groups and self.disc_selected) else None
        current = "" if info is None else ("Ignored" if info["status"] == "ignored" else info["name"])
        self.disc_name_combo.set_classes([(n, d["color"].name()) for n, d in self.classes.items()], current)
        self.disc_add_class_btn.hide()

    def _discover_name_typed(self, text):
        t = text.strip()
        self.disc_add_class_btn.setVisible(bool(t) and t not in self.classes and t not in ("Unnamed", "Ignored"))
        if self.disc_add_class_btn.isVisible():
            self.disc_name_feedback.setText(f"'{t}' is not a class yet. Press Add Class (or Enter) to create it.")
        else:
            self.disc_name_feedback.setText("")

    def _after_group_naming(self, message):
        self.disc_name_feedback.setText(message)
        print("Discover: " + message)
        self.refresh_discover_list()
        self._refresh_name_combo()
        self.disc_ignore_btn.setText("Restore Group" if self.groups.groups[self.disc_selected]["status"] == "ignored" else "Ignore Group")
        self._update_sel_header()
        self.update_discover_map()

    def _naming_targets(self):
        """ Groups a naming action applies to: all multi-selected cards, else the selected one. """
        if len(self.disc_multi) >= 2:
            return [k for k in self.disc_multi if k in self.groups.groups]
        return [self.disc_selected] if self.disc_selected else []

    def discover_assign_class(self, name):
        if self.groups is None or not self._naming_targets() or name not in self.classes:
            return
        keys = self._naming_targets()
        for k in keys:
            self.groups.set_name(k, name)
        self._after_group_naming(f"{', '.join(keys)} {'is' if len(keys) == 1 else 'are'} now '{name}'.")

    def discover_assign_special(self, kind):
        if self.groups is None or not self._naming_targets():
            return
        keys = self._naming_targets()
        for k in keys:
            if kind == "ignored":
                self.groups.set_ignored(k, True)
            else:
                self.groups.set_name(k, "")
        verb = "is" if len(keys) == 1 else "are"
        self._after_group_naming(f"{', '.join(keys)} {verb} {'ignored' if kind == 'ignored' else 'unnamed'}.")

    def discover_add_typed_class(self):
        if self.groups is None or self.disc_selected is None:
            return
        text = self.disc_name_combo.currentText().strip()
        if not text:
            return
        if text in ("Unnamed", "Ignored"):
            self.discover_assign_special(text.lower())
            return
        if text not in self.classes:
            self._ensure_class(text)
            keys = self._naming_targets()
            for k in keys:
                self.groups.set_name(k, text)
            self._after_group_naming(f"Class '{text}' added and given to {', '.join(keys)}.")
        else:
            self.discover_assign_class(text)

    def discover_rename_class(self, name):
        new, ok = QInputDialog.getText(self, "Rename Class", f"New name for '{name}':", text=name)
        if ok and self._rename_class(name, new):
            self.disc_name_feedback.setText(f"Class '{name}' renamed to '{new.strip()}'.")

    def discover_recolor_class(self, name):
        color = QColorDialog.getColor(self.classes[name]["color"], self, f"Colour for '{name}'")
        if color.isValid():
            self._set_class_color(name, color)
            self.disc_name_feedback.setText(f"Class '{name}' recoloured.")

    def discover_remove_class(self, name):
        if name not in self.classes:
            return
        n_groups = sum(1 for v in self.groups.groups.values() if v["name"] == name) if self.groups else 0
        n_samples = len(self.classes[name]["training_segments"])
        detail = [f"{n_groups} group(s) named '{name}' become unnamed."]
        if n_samples:
            detail.append(f"Its {n_samples} training sample(s) in Classification are removed too.")
        if self.raw_classified_raster_ints is not None:
            detail.append("Areas mapped as this class become unclassified.")
        reply = QMessageBox.question(self, "Remove Class", f"Remove class '{name}'?\n\n" + "\n".join(detail),
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        self._delete_class(name)
        if self.groups is not None and self.disc_selected is not None:
            self._after_group_naming(f"Class '{name}' removed.")

    def discover_toggle_ignore(self):
        if self.groups is None or self.disc_selected is None:
            return
        key = self.disc_selected
        ignored = self.groups.groups[key]["status"] != "ignored"
        self.groups.set_ignored(key, ignored)
        print(f"Discover: {key} {'ignored' if ignored else 'restored'}.")
        self.refresh_discover_list()
        self.show_discover_group(key)

    def discover_merge(self):
        if self.groups is None:
            return
        g = self.groups
        if len(self.disc_multi) >= 2:
            keys = list(self.disc_multi)
            reply = QMessageBox.question(self, "Merge Groups", f"Merge {', '.join(keys)} into one group?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply != QMessageBox.Yes:
                return
            sets = [keys]
        else:
            sets = self._merge_suggestion_dialog()
            if not sets:
                return
        kept = []
        for keys in sets:
            keys = [k for k in keys if k in g.groups]
            keep = g.merge(keys)
            if keep:
                kept.append(keep)
                for k in keys:
                    self._strip_cache.pop(k, None)
                print(f"Discover: merged {', '.join(keys)} into {keep}.")
        self.disc_multi = []
        self._sim_space = None
        self.refresh_discover_list()
        self.show_discover_group(kept[0] if kept else self.disc_selected)

    def _merge_suggestion_dialog(self):
        """ Suggested look-alike pairs with chip strips; returns the ticked sets (pairs that
        share a group are joined into one set). """
        g = self.groups
        sugg = g.merge_suggestions(8)
        if self.disc_selected and self.disc_selected in g.groups:
            for c in g.closest_groups(self.disc_selected, 2):
                if not any({a, b} == {self.disc_selected, c} for a, b, _ in sugg):
                    sugg.insert(0, (self.disc_selected, c, 0.0))
        dlg = QDialog(self)
        dlg.setWindowTitle("Merge Groups")
        lay = QVBoxLayout(dlg)
        intro = QLabel("Pairs of groups whose centres are closest. Tick the pairs that really are the same thing.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        host = QWidget()
        hl = QVBoxLayout(host)
        boxes = []
        for a, b, _ in sugg:
            frame = QGroupBox()
            fl = QVBoxLayout(frame)
            cb = QCheckBox(f"{a} + {b}")
            fl.addWidget(cb)
            for k in (a, b):
                row = QHBoxLayout()
                lbl = QLabel(k + (f" · {g.groups[k]['name']}" if g.groups[k]["name"] else ""))
                lbl.setFixedWidth(120)
                row.addWidget(lbl)
                strip = QLabel()
                strip.setPixmap(self._group_strip(k))
                row.addWidget(strip, 1)
                fl.addLayout(row)
            hl.addWidget(frame)
            boxes.append((cb, a, b))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(host)
        scroll.setMinimumSize(520, 460)
        lay.addWidget(scroll, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        ok = QPushButton("Merge Ticked")
        ok.clicked.connect(dlg.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(dlg.reject)
        row.addWidget(ok)
        row.addWidget(cancel)
        lay.addLayout(row)
        if dlg.exec() != QDialog.Accepted:
            return []
        sets = []
        for cb, a, b in boxes:
            if not cb.isChecked():
                continue
            hit = [s for s in sets if a in s or b in s]
            merged = {a, b}.union(*hit) if hit else {a, b}
            sets = [s for s in sets if s not in hit] + [merged]
        return [sorted(s) for s in sets]

    def discover_split(self):
        if self.groups is None or self.disc_selected is None:
            return
        g, key = self.groups, self.disc_selected
        fdf = self.engine.features_df.reindex(g.ids)
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Split Group {key}")
        lay = QVBoxLayout(dlg)
        dlg.resize(820, 520)
        lay.addWidget(QLabel(f"Re-group only the {g.size(key):,} segments of {key}."))
        row = QHBoxLayout()
        row.addWidget(QLabel("Split into"))
        parts_combo = QComboBox()
        parts_combo.addItems(["2", "3", "4"])
        row.addWidget(parts_combo)
        row.addWidget(QLabel("groups, using"))
        row.addStretch(1)
        lay.addLayout(row)
        rb_app = QRadioButton("Appearance (the same features as Find Groups)")
        rb_height = QRadioButton("Height above surroundings (DSM)")
        rb_bright = QRadioButton("Brightness")
        rb_app.setChecked(True)
        has_height = "dsm_rel_neighbors" in fdf.columns and self.engine.has_dsm
        rb_height.setEnabled(has_height)
        for rb in (rb_app, rb_height, rb_bright):
            lay.addWidget(rb)
        lay.addWidget(QLabel("Preview"))
        preview_host = QWidget()
        preview_layout = QVBoxLayout(preview_host)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(preview_host)

        def values():
            if rb_height.isChecked():
                return fdf["dsm_rel_neighbors"].values
            if rb_bright.isChecked():
                return fdf["brightness_mean"].values
            return None

        def preview():
            while preview_layout.count():
                it = preview_layout.takeAt(0)
                if it.widget():
                    it.widget().deleteLater()
            import copy
            tmp = copy.copy(g)
            tmp.labels = g.labels.copy()
            tmp.groups = {k: dict(v) for k, v in g.groups.items()}
            new = tmp.split(key, int(parts_combo.currentText()), values())
            v = values()
            for nk in new:
                mem = tmp.members(nk)
                extra = ""
                if v is not None and len(mem):
                    unit = " m vs surroundings" if rb_height.isChecked() else " brightness"
                    extra = f" · mean {np.nanmean(v[mem]):.1f}{unit}"
                lbl = QLabel(f"<b>{nk}</b> · {len(mem):,} segments{extra} (scroll sideways to see them all)")
                preview_layout.addWidget(lbl)
                strip = ChipGallery(chip_size=72, strip=True, selectable=False)
                strip.set_segments([int(tmp.ids[i]) for i in tmp.typical(nk, len(mem))], self._chip, first_batch=24,
                                   tip_fn=self._tile_tip)
                preview_layout.addWidget(strip)

        for w in (rb_app, rb_height, rb_bright):
            w.toggled.connect(lambda on: on and preview())
        parts_combo.currentIndexChanged.connect(lambda _: preview())
        preview()
        brow = QHBoxLayout()
        brow.addStretch(1)
        ok = QPushButton("Apply Split")
        ok.clicked.connect(dlg.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(dlg.reject)
        brow.addWidget(ok)
        brow.addWidget(cancel)
        lay.addLayout(brow)
        if dlg.exec() != QDialog.Accepted:
            return
        new = g.split(key, int(parts_combo.currentText()), values())
        self._strip_cache.pop(key, None)
        print(f"Discover: split {key} into {', '.join(new)}.")
        self.refresh_discover_list()
        self.show_discover_group(new[0] if new else key)

    def discover_find_similar_elsewhere(self):
        if self.groups is None or self.disc_selected is None:
            return
        g, key = self.groups, self.disc_selected
        idx = g.similar(None, 25, exclude_key=key, query=g.centre(key))
        chips = [self._chip(g.ids[i], 120) for i in idx]
        caps = [f"from {g.labels[i] or 'edge'}" for i in idx]
        dlg = ChipPickerDialog(
            f"Segments like {key}", f"The 25 segments outside {key} that look most like it, closest first. "
            f"Untick any that don't belong, then move the rest into {key}.",
            chips, caps, [key], "Move ticked into", "Move", parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        picked = [idx[i] for i in dlg.ticked_indices()]
        if not picked:
            return
        for i in picked:
            self._strip_cache.pop(g.labels[i], None)
        g.move(picked, key)
        self._strip_cache.pop(key, None)
        print(f"Discover: moved {len(picked)} segment(s) into {key}.")
        self.refresh_discover_list()
        self.show_discover_group(key)

    # --- Discover: map ---
    def on_discover_map_clicked(self, pos):
        if self.groups is None:
            return
        sid = self._segment_at(pos)
        i = self.groups.pos.get(sid)
        if i is None or not self.groups.labels[i]:
            return
        key = self.groups.labels[i]
        if self.disc_filter_combo.currentText() != "All groups" and key not in self._visible_keys():
            self.disc_filter_combo.setCurrentIndex(0)
        self.disc_multi = []
        self.show_discover_group(key)

    def update_discover_map(self):
        """ Group overlay as an array layer at display resolution over the shared ortho
        pixmap (no full-size copy), plus optional segment edges and the zoomed-to segment. """
        if self.groups is None or self.raw_pixmap is None or self.engine.segmented_image is None:
            return
        g = self.groups
        v = self.discover_viewer
        v.set_base_image_cached(self.raw_pixmap, grid_size=self.raw_grid)
        st, seg_d = self._seg_disp()
        lut = np.zeros((int(self.engine.segmented_image.max()) + 1, 4), dtype=np.uint8)
        mode = "selected" if self.disc_mode_selected.isChecked() else ("all" if self.disc_mode_all.isChecked() else "named")
        for k, info in g.groups.items():
            mem = g.ids[g.labels == k]
            if not len(mem):
                continue
            c = QColor(self._group_color(k))
            if mode == "all":
                rgba = rgba_premultiplied(QColor(120, 120, 120), 110) if info["status"] == "ignored" else rgba_premultiplied(c, 200)
            elif mode == "named":
                if info["status"] != "named":
                    continue
                rgba = rgba_premultiplied(c, 210)
            else:
                if k == self.disc_selected:
                    rgba = rgba_premultiplied(c, 230)
                elif k in self.disc_multi:
                    rgba = rgba_premultiplied(QColor(241, 213, 143), 190)
                else:
                    continue
            lut[mem] = rgba
        arr = lut[seg_d]
        if mode == "selected" and self.disc_selected is not None:
            m = arr[..., 3] > 0
            edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
            arr[edge] = (255, 255, 255, 255)
        v.set_array_layer("groups", arr, z=1)
        v.set_layer("edges", self.boundary_overlay_qimg, 1.0, z=2)
        self._update_discover_focus()
        self._apply_discover_layer_visibility()
        label = {"selected": f"{self.disc_selected or '—'} highlighted", "all": "all groups", "named": "named groups"}[mode]
        self.disc_map_label.setText(f"View: {label} ·")

    def _apply_discover_layer_visibility(self):
        v = self.discover_viewer
        v.set_layer_opacity("groups", self.disc_opacity_slider.value() / 100.0 if self.disc_overlay_check.isChecked() else 0.0)
        v.set_layer_opacity("edges", 1.0 if self.disc_edges_check.isChecked() else 0.0)

    def _update_discover_focus(self):
        """ Cyan outline of the segment last zoomed to (full resolution, its own small layer). """
        sid = getattr(self, "disc_focus_seg", None)
        sl = self._seg_slices() if sid else None
        if not sid or sid > len(sl) or sl[sid - 1] is None:
            self.discover_viewer.set_layer("focus", None)
            return
        rs, cs = sl[sid - 1]
        r0, r1 = max(0, rs.start - 2), min(self.engine.segmented_image.shape[0], rs.stop + 2)
        c0, c1 = max(0, cs.start - 2), min(self.engine.segmented_image.shape[1], cs.stop + 2)
        m = self.engine.segmented_image[r0:r1, c0:c1] == sid
        inner = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
        edge = m & ~inner
        for _ in range(2):
            edge = edge | np.roll(edge, 1, 0) | np.roll(edge, 1, 1)
        img = np.zeros(m.shape + (4,), dtype=np.uint8)
        img[edge] = rgba_premultiplied(QColor(0, 229, 255), 255)
        h, w = m.shape
        q = QImage(img.data, w, h, 4 * w, QImage.Format_ARGB32_Premultiplied).copy()
        self.discover_viewer.set_layer("focus", q, 1.0, z=5, rect=(c0, r0, w, h))

    def on_discover_tile_double_clicked(self, item):
        """ Zooms the map to the tile's segment (about 30 m across) and outlines it. """
        sid = int(item.data(Qt.UserRole))
        ids, cents, pos, _ = self._segment_index()
        if sid not in pos:
            return
        r, c = cents[pos[sid]]
        px = self.engine.effective_pixel_size or 0.1
        zoom = max(0.05, self.discover_viewer.viewport().width() / (30.0 / px))
        self.disc_focus_seg = sid
        self._update_discover_focus()
        self.discover_viewer.center_on_pixmap_point(float(c), float(r), zoom=zoom)

    def on_discover_map_context(self, pixmap_pos, global_pos):
        if self.groups is None:
            return
        sid = self._segment_at(pixmap_pos)
        i = self.groups.pos.get(sid)
        if i is None:
            return
        g = self.groups
        cur = g.labels[i]
        menu = QMenu(self)
        head = menu.addAction(f"Segment #{sid} · " + ((f"{cur}" + (f" ({g.groups[cur]['name']})" if g.groups[cur]["name"] else "")) if cur else "no group"))
        head.setEnabled(False)
        menu.addSeparator()
        sub = menu.addMenu("Move Segment to Group")
        for k in [k for k in g.keys() if g.size(k) > 0 and k != cur]:
            name = g.groups[k]["name"]
            act = sub.addAction(k + (f" · {name}" if name else "") + ("  (ignored)" if g.groups[k]["status"] == "ignored" else ""))
            act.triggered.connect(lambda _=False, k=k: self._move_tiles([sid], k))
        menu.addAction("Move Segment to a New Group", lambda: self._move_tiles([sid], g.new_group()))
        if cur:
            menu.addAction("Leave Segment Out of All Groups", lambda: self._move_tiles([sid], ""))
            menu.addSeparator()
            menu.addAction(f"Show Its Group ({cur})", lambda: (self.disc_filter_combo.setCurrentIndex(0), self.show_discover_group(cur)))
        menu.exec(global_pos)

    # --- Discover: to Classification ---
    def _named_groups_by_class(self):
        out = {}
        for k, info in self.groups.groups.items():
            if info["status"] == "named" and self.groups.size(k) > 0:
                out.setdefault(info["name"], []).append(k)
        return out

    def _ensure_class(self, name, color_hex=None):
        if name in self.classes:
            return
        color = QColor(suggest_class_color(name, [d["color"].name() for d in self.classes.values()]))
        self.classes[name] = {"color": color, "training_segments": set()}
        item = QListWidgetItem(f"{name} (0)")
        item.setBackground(color)
        self.class_list_widget.addItem(item)
        print(f" -> New class '{name}' created from Discover.")

    def send_groups_to_classification(self):
        if self.groups is None:
            return
        by_class = self._named_groups_by_class()
        if not by_class:
            print("Discover: name at least one group first.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Send Named Groups to Classification")
        lay = QVBoxLayout(dlg)
        intro = QLabel("Each named group gives its most typical segments to its class as training "
                       "samples. Unnamed and ignored groups are not used.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        table = QTableWidget(len(by_class), 3)
        table.setHorizontalHeaderLabels(["Class", "From groups", "Samples"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        n_row = QHBoxLayout()
        n_row.addWidget(QLabel("Typical segments per group:"))
        per_spin = QSpinBox()
        per_spin.setRange(1, 200)
        per_spin.setValue(12)
        n_row.addWidget(per_spin)
        n_row.addStretch(1)

        def fill():
            for r, (cls, keys) in enumerate(by_class.items()):
                table.setItem(r, 0, QTableWidgetItem(cls + ("" if cls in self.classes else "  (new class)")))
                table.setItem(r, 1, QTableWidgetItem(", ".join(keys)))
                n = sum(min(per_spin.value(), self.groups.size(k)) for k in keys)
                table.setItem(r, 2, QTableWidgetItem(str(n)))
        fill()
        per_spin.valueChanged.connect(lambda _: fill())
        lay.addWidget(table)
        lay.addLayout(n_row)
        rb_add = QRadioButton("Add to existing training samples")
        rb_replace = QRadioButton("Replace the existing samples of these classes")
        rb_add.setChecked(True)
        lay.addWidget(rb_add)
        lay.addWidget(rb_replace)
        brow = QHBoxLayout()
        brow.addStretch(1)
        ok = QPushButton("Send and Open Classification")
        ok.clicked.connect(dlg.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(dlg.reject)
        brow.addWidget(ok)
        brow.addWidget(cancel)
        lay.addLayout(brow)
        dlg.resize(560, 420)
        if dlg.exec() != QDialog.Accepted:
            return
        per = per_spin.value()
        total = 0
        for cls, keys in by_class.items():
            self._ensure_class(cls, self.groups.groups[keys[0]]["color"])
            if rb_replace.isChecked():
                self.classes[cls]["training_segments"] = set()
            for k in keys:
                sids = {int(self.groups.ids[i]) for i in self.groups.typical(k, per)}
                for other in self.classes.values():
                    other["training_segments"] -= sids
                self.classes[cls]["training_segments"] |= sids
                total += len(sids)
        self.rebuild_sample_class_map()
        self.update_list_widget_counts()
        print(f"Discover: sent {total} training samples to {len(by_class)} class(es): {', '.join(by_class)}.")
        self.tabs.setCurrentWidget(self.classification_tab)
        self.update_classification_display()

    # --- Discover: workspace persistence ---
    def _discover_save_to_workspace(self):
        if self.deep is not None:
            od.save_deep_features(os.path.join(self.workspace_dir, "deep_features.npz"), self.deep)
        if self.groups is not None:
            with open(os.path.join(self.workspace_dir, "groups.json"), "w") as f:
                json.dump(self.groups.to_json(), f, indent=1)

    def _discover_load_from_workspace(self):
        self._discover_reset()
        dpath = os.path.join(self.workspace_dir, "deep_features.npz")
        gpath = os.path.join(self.workspace_dir, "groups.json")
        if os.path.exists(dpath):
            try:
                self.deep = od.load_deep_features(dpath)
                print(" -> Deep features restored from the workspace cache.")
            except Exception as e:
                print(f" -> Deep feature cache unreadable ({e}); it will be recomputed if needed.")
        if os.path.exists(gpath):
            try:
                with open(gpath) as f:
                    data = json.load(f)
            except Exception as e:
                print(f" -> Saved groups could not be read ({e}).")
                return
            self._discover_restore_groups(data, "restored from the workspace")

    def _discover_restore_groups(self, data, source):
        """ Rebuilds self.groups from GroupModel.to_json() output (groups.json, or an
        iteration snapshot). data=None clears the groups but keeps the deep features.
        Returns True if groups are in place afterwards. """
        self.groups = None
        self.disc_selected = None
        self.disc_multi = []
        self.disc_chip_offset = 0
        self._strip_cache = {}
        self._disc_cards = {}
        try:
            if data is None:
                return False
            st = data.get("settings", {})
            if st.get("fingerprint") != self._segment_index()[3]:
                print(" -> Saved groups belong to a different segmentation; press Find Groups to rebuild.")
                return False
            deep = self._deep_for_current() if st.get("use_deep") else None
            if st.get("use_deep") and deep is None:
                print(" -> Saved groups need deep features that are missing; press Find Groups to rebuild.")
                return False
            ids = np.array([s for s in self._segment_index()[0] if s in self.engine.features_df.index])
            X = od.build_space(ids, self.engine.features_df, self.engine.active_feature_columns(include_deep=False),
                               deep, bool(st.get("use_deep")), bool(st.get("use_features", True)))
            self.groups = od.GroupModel.from_json(data, ids, X)
            self.disc_use_deep_check.setChecked(bool(st.get("use_deep")))
            self.disc_use_feat_check.setChecked(bool(st.get("use_features", True)))
            self.disc_n_spin.setValue(int(st.get("n_groups", 30)))
            self._discover_set_enabled(True)
            self.refresh_discover_list()
            self.discover_viewer.set_image(self.raw_pixmap, grid_size=self.raw_grid)
            keys = self._visible_keys()
            self.show_discover_group(keys[0] if keys else None)
            named = sum(1 for v in self.groups.groups.values() if v["status"] == "named")
            self.disc_status_label.setText(f"{len(ids):,} segments · {len(self.groups.groups)} groups "
                                           f"({named} named), {source}")
            print(f" -> {len(self.groups.groups)} Discover groups {source} ({named} named).")
            return True
        except Exception as e:
            self.groups = None
            print(f" -> Saved groups could not be restored ({e}).")
            return False
        finally:
            if self.groups is None:
                self.refresh_discover_list()
                self.show_discover_group(None)
                self._discover_set_enabled(False)
                if self.engine.segmented_image is not None:
                    self.disc_status_label.setText("Ready: press Find Groups.")

    # --- TAB 3: Classification ---
    def init_classification_tab(self):
        tab3 = QWidget()
        self.classification_tab = tab3
        layout = QHBoxLayout(tab3)

        col1_widget = QWidget()
        col1_layout = QVBoxLayout(col1_widget)
        col1_widget.setFixedWidth(self.LEFT_COL1_W)

        col2_widget = QWidget()
        col2_layout = QVBoxLayout(col2_widget)
        col2_widget.setFixedWidth(self.LEFT_COL2_W)

        # Workspace load/save live in the File menu (init_menu_bar).

        # DSM status realignment widget (Tab 2 specific reload singly / unload)
        dsm_status_group = QGroupBox("DSM Controls")
        dsm_status_layout = QVBoxLayout()
        self.dsm_status_label = QLabel("DSM Status: INACTIVE (Spectral maps only)")
        self.dsm_status_label.setWordWrap(True)

        dsm_action_layout = QHBoxLayout()
        reload_dsm_btn = QPushButton("Reload DSM Singly")
        reload_dsm_btn.clicked.connect(self.reload_dsm_singly)
        unload_dsm_btn = QPushButton("Unload DSM")
        unload_dsm_btn.clicked.connect(self.unload_surface_data)
        dsm_action_layout.addWidget(reload_dsm_btn)
        dsm_action_layout.addWidget(unload_dsm_btn)

        dsm_status_layout.addWidget(self.dsm_status_label)
        dsm_status_layout.addLayout(dsm_action_layout)
        dsm_status_group.setLayout(dsm_status_layout)

        # --- Interactive Active Views: Show Training Samples toggle now lives here
        # (moved off Class Management), and the view-mode buttons below have fixed,
        # spec'd toggle logic -- see toggle_segmented_view/toggle_classified_view/
        # toggle_blend_mode/_sync_blend_availability for the actual state machine.
        view_group = QGroupBox("Interactive Active Views")
        view_layout = QVBoxLayout()

        samples_row = QHBoxLayout()
        samples_row.addWidget(QLabel("Show Training Samples"))
        self.toggle_samples_btn = ToggleSwitch()
        self.toggle_samples_btn.setChecked(True)
        self.toggle_samples_btn.clicked.connect(self.toggle_samples_view)
        samples_row.addWidget(self.toggle_samples_btn)
        samples_row.addStretch()
        view_layout.addLayout(samples_row)

        btn_layout = QHBoxLayout()
        self.blend_toggle_btn = QPushButton("Blend Overlay")
        self.blend_toggle_btn.setCheckable(True)
        self.blend_toggle_btn.setEnabled(False)  # no classification yet -- nothing to blend
        self.blend_toggle_btn.clicked.connect(self.toggle_blend_mode)

        self.seg_view_btn = QPushButton("Segmented")
        self.seg_view_btn.setCheckable(True)
        self.seg_view_btn.setChecked(True)
        self.seg_view_btn.clicked.connect(self.toggle_segmented_view)

        self.cls_view_btn = QPushButton("Raw Ortho")
        self.cls_view_btn.setCheckable(True)
        self.cls_view_btn.clicked.connect(self.toggle_classified_view)

        btn_layout.addWidget(self.blend_toggle_btn)
        btn_layout.addWidget(self.seg_view_btn)
        btn_layout.addWidget(self.cls_view_btn)
        view_layout.addLayout(btn_layout)

        # --- Making the Opacity Slider permanently visible ---
        self.opacity_container = QWidget()
        opacity_lay = QHBoxLayout(self.opacity_container)
        opacity_lay.addWidget(QLabel("Opacity:"))

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setEnabled(False) # Default disabled until blend mode active

        # Applies when the slider is released (setTracking(False)), not on every step.
        self.opacity_slider.setTracking(False)
        self.opacity_slider.valueChanged.connect(self.update_classification_display)
        # On/off switch: hides the classified layer (opacity 0) or restores the slider value.
        self.opacity_toggle = ToggleSwitch()
        self.opacity_toggle.setChecked(True)
        self.opacity_toggle.setEnabled(False)
        self.opacity_toggle.setToolTip("Show / hide the classified layer over the ortho")
        self.opacity_toggle.clicked.connect(self.update_classification_display)
        # Kept for code that still refers to it; hidden.
        self.opacity_spinbox = self.opacity_toggle

        # How the classified layer mixes with the ortho in Blend Overlay: Multiply tints the
        # ortho (texture stays visible), Normal is plain transparency (truer colours).
        self.blend_mode_combo = QComboBox()
        self.blend_mode_combo.addItems(["Multiply", "Normal"])
        self.blend_mode_combo.setToolTip("Multiply: the class colour tints the ortho, keeping its texture. "
                                         "Normal: plain transparency -- truer colours, less texture.")
        self.blend_mode_combo.setEnabled(False)
        self.blend_mode_combo.currentIndexChanged.connect(lambda _: self.update_classification_display())

        opacity_lay.addWidget(self.opacity_slider)
        opacity_lay.addWidget(self.blend_mode_combo)
        opacity_lay.addWidget(self.opacity_toggle)
        view_layout.addWidget(self.opacity_container)
        view_group.setLayout(view_layout)

        gen_group = QGroupBox("Post-Classification Generalization")
        gen_layout = QHBoxLayout()
        self.gen_slider = QSlider(Qt.Horizontal)
        self.gen_slider.setRange(0, 2000)
        self.gen_slider.setValue(0)
        self.gen_spinbox = QSpinBox()
        self.gen_spinbox.setRange(0, 2000)
        self.gen_spinbox.setValue(0)
        self.gen_slider.valueChanged.connect(self.gen_spinbox.setValue)
        self.gen_spinbox.valueChanged.connect(self.gen_slider.setValue)
        self.gen_slider.sliderReleased.connect(self.apply_generalization_sieve)
        self.gen_spinbox.editingFinished.connect(self.apply_generalization_sieve)
        # The same threshold as an area on the ground (pixels x pixel area), kept in step
        # with the pixel box; see _sync_mmu_m2.
        self.gen_spinbox.setSuffix(" px")
        self.gen_m2_spinbox = QDoubleSpinBox()
        self.gen_m2_spinbox.setDecimals(1)
        self.gen_m2_spinbox.setSuffix(" m²")
        self.gen_m2_spinbox.setRange(0.0, 0.0)
        self.gen_m2_spinbox.setToolTip("Minimum mapping unit as an area on the ground: patches of one "
                                       "class smaller than this join the class around them.")
        self.gen_spinbox.valueChanged.connect(self._mmu_px_changed)
        self.gen_m2_spinbox.valueChanged.connect(self._mmu_m2_changed)
        self.gen_m2_spinbox.editingFinished.connect(self.apply_generalization_sieve)
        self.gen_m2_label = QLabel("=")
        self.gen_m2_label.setVisible(False)       # shown once an image in projected coordinates is loaded
        self.gen_m2_spinbox.setVisible(False)
        gen_layout.addWidget(QLabel("MMU:"))
        gen_layout.addWidget(self.gen_slider)
        gen_layout.addWidget(self.gen_spinbox)
        gen_layout.addWidget(self.gen_m2_label)
        gen_layout.addWidget(self.gen_m2_spinbox)
        gen_group.setLayout(gen_layout)

        # --- Class Management: name input moved to the top; Merge/Remove near the
        # list, Clear Selected/Clear All pinned at the bottom with confirmation
        # popups since all four are destructive.
        class_group = QGroupBox("Class Management")
        class_layout = QVBoxLayout()

        add_class_layout = QHBoxLayout()
        self.class_name_edit = QLineEdit()
        self.class_name_edit.setPlaceholderText("Enter new class name...")
        add_class_btn = QPushButton("Add")
        add_class_btn.clicked.connect(self.add_class)
        add_class_layout.addWidget(self.class_name_edit)
        add_class_layout.addWidget(add_class_btn)
        class_layout.addLayout(add_class_layout)

        self.class_list_widget = QListWidget()
        self.class_list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        self.class_list_widget.setItemDelegate(ClassListDelegate(self.class_list_widget))
        self.class_list_widget.itemSelectionChanged.connect(self.on_class_selection_changed)
        self.class_list_widget.itemDoubleClicked.connect(self.edit_class_dialog)
        class_layout.addWidget(self.class_list_widget, 1)

        self.merge_classes_btn = QPushButton("Merge Selected Classes")
        self.merge_classes_btn.clicked.connect(self.merge_selected_classes)
        class_layout.addWidget(self.merge_classes_btn)

        remove_class_btn = QPushButton("Remove Selected Class")
        remove_class_btn.clicked.connect(self.remove_class)
        class_layout.addWidget(remove_class_btn)

        class_layout.addSpacing(10)

        clear_selected_class_btn = QPushButton("Clear Selected Class")
        clear_selected_class_btn.clicked.connect(self.clear_selected_class_training_data)
        class_layout.addWidget(clear_selected_class_btn)

        clear_all_training_btn = QPushButton("Clear All Training Data")
        clear_all_training_btn.clicked.connect(self.clear_training_data)
        class_layout.addWidget(clear_all_training_btn)

        class_group.setLayout(class_layout)

        # --- Training & Classification: Train + Classify side by side and larger;
        # Export pulled out on its own below (see export_btn further down).
        train_group = QGroupBox("Training & Classification")
        train_outer = QVBoxLayout()
        separability_btn = QPushButton("Check Class Separability")
        separability_btn.setToolTip("How distinct the classes' training samples are, before training "
                                    "(Jeffries-Matusita distance and signature plot)")
        separability_btn.clicked.connect(self.check_class_separability)
        label_next_btn = QPushButton("What to Label Next…")
        label_next_btn.setToolTip("Segments the trained classifier is least sure about, spread over different "
                                  "kinds of segment -- labelling these teaches it the most")
        label_next_btn.clicked.connect(self.show_label_next)
        helper_row = QHBoxLayout()
        helper_row.addWidget(separability_btn)
        helper_row.addWidget(label_next_btn)
        train_outer.addLayout(helper_row)
        self.use_deep_check = QCheckBox("Use deep appearance features")
        self.use_deep_check.setToolTip(
            "Also give the classifier 20 appearance features from the DINOv2 deep features (10 principal "
            "components each of the 5 m and 10 m patches around every segment). Uses the deep features "
            "computed in Discover Classes (Find Groups with Appearance ticked).")
        train_outer.addWidget(self.use_deep_check)
        train_layout = QHBoxLayout()
        train_btn = QPushButton("Train Classifier")
        train_btn.setMinimumHeight(44)
        train_btn.clicked.connect(self.train_classifier)
        classify_btn = QPushButton("Classify All Segments")
        classify_btn.setMinimumHeight(44)
        classify_btn.clicked.connect(self.classify_all)
        train_layout.addWidget(train_btn)
        train_layout.addWidget(classify_btn)
        train_outer.addLayout(train_layout)
        train_group.setLayout(train_outer)

        self.export_btn = QPushButton("Export Final Classified TIF")
        self.export_btn.setMinimumHeight(48)
        self.export_btn.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: white; font-weight: 600; border-radius: 4px; }"
            "QPushButton:hover { background-color: #a93226; }"
            "QPushButton:disabled { background-color: #d3a29c; color: #f2e6e4; }"
        )
        self.export_btn.clicked.connect(self.export_classified_tif)


        selection_group = QGroupBox("Selection & Merge Tools")
        selection_layout = QVBoxLayout()

        self.selection_mode_btn = QPushButton("Selection Mode: OFF")
        self.selection_mode_btn.setCheckable(True)
        self.selection_mode_btn.clicked.connect(self.toggle_selection_mode)
        selection_layout.addWidget(self.selection_mode_btn)

        self.selection_count_label = QLabel("Selected: 0 segments")
        selection_layout.addWidget(self.selection_count_label)

        tol_layout = QHBoxLayout()
        self.grow_tolerance_slider = QSlider(Qt.Horizontal)
        self.grow_tolerance_slider.setRange(0, 300)
        self.grow_tolerance_slider.setValue(100)
        self.grow_tolerance_spinbox = QDoubleSpinBox()
        self.grow_tolerance_spinbox.setRange(0.0, 3.0)
        self.grow_tolerance_spinbox.setSingleStep(0.05)
        self.grow_tolerance_spinbox.setValue(1.0)
        self.grow_tolerance_slider.valueChanged.connect(lambda v: self.grow_tolerance_spinbox.setValue(v / 100.0))
        self.grow_tolerance_spinbox.valueChanged.connect(lambda v: self.grow_tolerance_slider.setValue(int(v * 100)))
        self.grow_tolerance_spinbox.valueChanged.connect(self.on_grow_tolerance_changed)
        tol_layout.addWidget(QLabel("Grow Tolerance:"))
        tol_layout.addWidget(self.grow_tolerance_slider)
        tol_layout.addWidget(self.grow_tolerance_spinbox)
        selection_layout.addLayout(tol_layout)

        clear_sel_btn = QPushButton("Clear Selection")
        clear_sel_btn.clicked.connect(self.clear_segment_selection)
        selection_layout.addWidget(clear_sel_btn)

        assign_sel_btn = QPushButton("Assign Selection to Active Class")
        assign_sel_btn.clicked.connect(self.assign_selection_to_class)
        selection_layout.addWidget(assign_sel_btn)

        merge_sel_btn = QPushButton("Merge Selection into Object")
        merge_sel_btn.clicked.connect(self.merge_selection_into_object)
        selection_layout.addWidget(merge_sel_btn)

        selection_group.setLayout(selection_layout)

        # Column 1: session management, DSM controls, and class management (expands vertically)
        col1_layout.addWidget(dsm_status_group)
        col1_layout.addWidget(class_group, 1)

        # Column 2: view controls, selection/merge tools, training/classification,
        # generalization, then Export and Reset pinned near the bottom -- Export right
        # under Generalization (it's the natural next step), Reset pushed to the very
        # bottom by the stretch since it's a "start over", not part of the main flow.
        col2_layout.addWidget(view_group)
        col2_layout.addWidget(selection_group)
        col2_layout.addWidget(train_group)
        col2_layout.addWidget(gen_group)
        col2_layout.addWidget(self.export_btn)
        col2_layout.addStretch()

        self.class_viewer = InteractivePhotoViewer()
        self.class_viewer.set_detail_provider(self._detail_image)
        self.class_viewer.photoClicked.connect(self.collect_training_sample)
        self.class_viewer.paintDragSegment.connect(self.paint_training_sample)
        self.class_viewer.paintDragFinished.connect(self.paint_drag_finished)
        self.class_viewer.segmentSelected.connect(self.handle_selection_click)
        self.class_viewer.selectDragSegment.connect(self.paint_selection_sample)
        self.class_viewer.selectDragFinished.connect(self.selection_drag_finished)
        self.class_viewer.undoRequested.connect(self.undo_selection)
        self.class_viewer.contextMenuRequested.connect(self.on_class_viewer_context_menu)
        
        class_map = QWidget()
        class_map_layout = QVBoxLayout(class_map)
        class_map_layout.setContentsMargins(0, 0, 0, 0)
        class_bar = QHBoxLayout()
        class_bar.addWidget(self._new_map_info(), 1)
        class_map_layout.addWidget(self._map_header([class_bar]))
        class_map_layout.addWidget(self.class_viewer, 1)
        class_map_layout.addWidget(self._map_tip("Click a segment to add it to the active class; Ctrl+drag to paint; "
                                                 "right-click for similar segments."))
        layout.addWidget(col1_widget)
        layout.addWidget(col2_widget)
        layout.addWidget(class_map, 1)
        self.tabs.addTab(tab3, "3. Classification")

    # --- TAB 4: Accuracy Assessment ---
    def init_accuracy_tab(self):
        """ Independent check of the aerial-ortho classification's own accuracy via an
        independent reference sample and a formal error matrix (Congalton & Green /
        Stehman convention). Two-column controls layout, same convention as
        init_classification_tab (col1/col2 QWidgets, not a QScrollArea) -- col1 holds
        Ground Truth Source + Classes (the tally table expands to fill remaining
        vertical space), col2 holds the point-generation/labeling/run workflow. The
        viewer here (self.acc_viewer) is deliberately its own InteractivePhotoViewer
        instance, always showing the raw/color-corrected ortho with no classification
        overlay -- showing the model's own prediction while someone is independently
        judging the true class would defeat the point of the check (anchoring bias). """
        tab_acc = QWidget()
        self.acc_tab_widget = tab_acc  # kept for focus-restoration checks elsewhere (see update_accuracy_point_list)
        layout = QHBoxLayout(tab_acc)

        col1_widget = QWidget()
        col1_layout = QVBoxLayout(col1_widget)
        col1_widget.setFixedWidth(self.LEFT_COL1_W)

        col2_widget = QWidget()
        col2_layout = QVBoxLayout(col2_widget)
        col2_widget.setFixedWidth(self.LEFT_COL2_W)

        # --- Column 1: Ground Truth Source, then Classes (expands to fill the rest) ---
        gt_group = QGroupBox("Ground Truth Source")
        gt_layout = QVBoxLayout()

        self.acc_gt_csv_radio = QRadioButton("Import Ground Truth CSV")
        self.acc_gt_expert_radio = QRadioButton("No Ground Truth Available (Generate Expert Points)")
        self.acc_gt_expert_radio.setChecked(True)
        self.acc_gt_btn_group = QButtonGroup(self)
        self.acc_gt_btn_group.addButton(self.acc_gt_csv_radio)
        self.acc_gt_btn_group.addButton(self.acc_gt_expert_radio)
        self.acc_gt_csv_radio.toggled.connect(self.sync_accuracy_gt_controls)
        gt_layout.addWidget(self.acc_gt_csv_radio)
        gt_layout.addWidget(self.acc_gt_expert_radio)

        gt_hint = QLabel(
            "Points are placed by simple random sampling and labeled by independent visual "
            "interpretation of the raw orthomosaic — ideally by someone other than whoever "
            "selected the training samples."
        )
        gt_hint.setWordWrap(True)
        gt_hint.setStyleSheet("color: #666; font-size: 10.5px;")
        gt_layout.addWidget(gt_hint)

        self.acc_csv_path_edit = QLineEdit()
        self.acc_csv_path_edit.setPlaceholderText("Path to ground-truth CSV...")
        self.acc_csv_path_edit.setReadOnly(True)
        csv_browse_btn = QPushButton("Browse")
        csv_browse_btn.clicked.connect(self.browse_for_accuracy_csv)
        csv_path_row = QHBoxLayout()
        csv_path_row.addWidget(self.acc_csv_path_edit)
        csv_path_row.addWidget(csv_browse_btn)
        self.acc_csv_path_row_widget = QWidget()
        self.acc_csv_path_row_widget.setLayout(csv_path_row)
        gt_layout.addWidget(self.acc_csv_path_row_widget)

        self.acc_supp_expert_check = QCheckBox("Also generate expert-visual points")
        self.acc_supp_expert_check.toggled.connect(self.sync_accuracy_gt_controls)
        gt_layout.addWidget(self.acc_supp_expert_check)

        self.acc_csv_summary_label = QLabel("")
        self.acc_csv_summary_label.setWordWrap(True)
        self.acc_csv_summary_label.setStyleSheet(
            "color: #333; font-size: 11px; border: 1px dashed #b7b7b7; border-radius: 3px; padding: 6px;"
        )
        gt_layout.addWidget(self.acc_csv_summary_label)

        gt_group.setLayout(gt_layout)
        col1_layout.addWidget(gt_group)

        # --- Classes (live tally, not a target tracker -- see plan doc). Given the
        # stretch factor below, this expands to fill all of column 1's remaining
        # height, matching Tab 2's Class Management group treatment. ---
        classes_group = QGroupBox("Classes")
        classes_layout = QVBoxLayout()
        self.acc_classes_table = QTableWidget(0, 2)
        self.acc_classes_table.setHorizontalHeaderLabels(["Class", "Points"])
        self.acc_classes_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.acc_classes_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.acc_classes_table.verticalHeader().setVisible(False)
        self.acc_classes_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.acc_classes_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.acc_classes_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.acc_classes_table.itemSelectionChanged.connect(self.on_accuracy_classes_table_selection_changed)
        classes_layout.addWidget(self.acc_classes_table)
        classes_group.setLayout(classes_layout)
        col1_layout.addWidget(classes_group, 1)

        # --- Iterations: one snapshot per Run Accuracy Assessment, so a round of
        # training/naming/points that made things worse can be rolled back. ---
        iter_group = QGroupBox("Iterations")
        iter_layout = QVBoxLayout()
        self.iter_table = QTableWidget(0, 4)
        self.iter_table.setHorizontalHeaderLabels(["#", "Overall Acc.", "Samples", "Points"])
        hh = self.iter_table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        self.iter_table.verticalHeader().setVisible(False)
        self.iter_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.iter_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.iter_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.iter_table.setToolTip(
            "Each Run Accuracy Assessment is kept here with the classes and training samples, Discover "
            "group names, accuracy points and classified map it was run on. Restore one to undo later "
            "changes that lowered accuracy. Double-click a row to see its results.")
        self.iter_table.itemSelectionChanged.connect(self._sync_iteration_buttons)
        self.iter_table.itemDoubleClicked.connect(lambda *_: self.view_iteration_results())
        iter_layout.addWidget(self.iter_table)
        iter_btn_row = QHBoxLayout()
        self.iter_restore_btn = QPushButton("Restore")
        self.iter_restore_btn.setToolTip("Return the training samples, class names, Discover group names, "
                                         "accuracy points and classified map to this iteration")
        self.iter_restore_btn.clicked.connect(self.restore_selected_iteration)
        self.iter_view_btn = QPushButton("Results")
        self.iter_view_btn.setToolTip("Show this iteration's error matrix")
        self.iter_view_btn.clicked.connect(self.view_iteration_results)
        self.iter_delete_btn = QPushButton("Delete")
        self.iter_delete_btn.setToolTip("Remove this iteration from the list")
        self.iter_delete_btn.clicked.connect(self.delete_selected_iteration)
        for b in (self.iter_restore_btn, self.iter_view_btn, self.iter_delete_btn):
            iter_btn_row.addWidget(b)
        iter_layout.addLayout(iter_btn_row)
        iter_group.setLayout(iter_layout)
        col1_layout.addWidget(iter_group, 1)

        # --- Column 2: save progress, point generation, labeling workspace, run ---
        self.acc_save_progress_btn = QPushButton("Save Points")
        self.acc_save_progress_btn.setToolTip(
            "Points here live in memory only until saved -- writes accuracy_points.gpkg to the "
            "workspace without re-writing segments/classification (that's what File > Save "
            "Workspace does). Points are restored automatically when you Load Workspace."
        )
        self.acc_save_progress_btn.clicked.connect(self.save_accuracy_progress)
        col2_layout.addWidget(self.acc_save_progress_btn)

        self.acc_gen_group = QGroupBox("Random Point Generation")
        gen_layout = QVBoxLayout()
        design_row = QHBoxLayout()
        design_row.addWidget(QLabel("Design:"))
        self.acc_design_combo = QComboBox()
        self.acc_design_combo.addItems(["Simple random", "Stratified by map class"])
        self.acc_design_combo.setToolTip(
            "Simple random: points anywhere in the image. Stratified: points drawn per class of the "
            "final classified map, each class getting at least the minimum below, so rare classes are "
            "assessed too. Stratified results are area-weighted (Olofsson et al. 2014; Stehman 2014) "
            "and include 95% confidence intervals and estimated class areas.")
        design_row.addWidget(self.acc_design_combo, 1)
        gen_layout.addLayout(design_row)
        self.acc_min_row = QWidget()
        min_layout = QHBoxLayout(self.acc_min_row)
        min_layout.setContentsMargins(0, 0, 0, 0)
        min_layout.addWidget(QLabel("Min per Class:"))
        self.acc_min_per_class_spinbox = QSpinBox()
        self.acc_min_per_class_spinbox.setRange(1, 1000)
        self.acc_min_per_class_spinbox.setValue(50)
        min_layout.addWidget(self.acc_min_per_class_spinbox)
        gen_layout.addWidget(self.acc_min_row)
        self.acc_min_row.setVisible(False)
        self.acc_design_combo.currentIndexChanged.connect(
            lambda i: self.acc_min_row.setVisible(i == 1))
        gen_row = QHBoxLayout()
        gen_row.addWidget(QLabel("Points to Add:"))
        self.acc_sample_size_spinbox = QSpinBox()
        self.acc_sample_size_spinbox.setRange(1, 20000)
        self.acc_sample_size_spinbox.setValue(200)
        gen_row.addWidget(self.acc_sample_size_spinbox)
        gen_points_btn = QPushButton("Generate")
        gen_points_btn.setToolTip("Add this many simple-random reference points")
        gen_points_btn.clicked.connect(self.generate_accuracy_points)
        gen_row.addWidget(gen_points_btn)
        gen_layout.addLayout(gen_row)
        self.acc_gen_group.setLayout(gen_layout)
        col2_layout.addWidget(self.acc_gen_group)

        # --- Point Assignment (the review/labeling workspace) ---
        assign_group = QGroupBox("Point Assignment")
        assign_layout = QVBoxLayout()

        filter_row = QHBoxLayout()
        self.acc_select_all_check = QCheckBox()
        self.acc_select_all_check.setToolTip("Select all / none (within the current filter)")
        self.acc_select_all_check.toggled.connect(self.on_accuracy_select_all_toggled)
        filter_row.addWidget(self.acc_select_all_check)
        self.acc_filter_combo = QComboBox()
        self.acc_filter_combo.addItem("All")
        self.acc_filter_combo.addItem("Unassigned")
        self.acc_filter_combo.currentTextChanged.connect(self.on_accuracy_filter_changed)
        filter_row.addWidget(self.acc_filter_combo, 1)
        clear_sel_btn = QPushButton("Clear Selection")
        clear_sel_btn.setToolTip("Deselect every checked point (across all filters), without changing their class assignments.")
        clear_sel_btn.clicked.connect(self.clear_accuracy_selection)
        filter_row.addWidget(clear_sel_btn)
        assign_layout.addLayout(filter_row)

        self.acc_point_list = QListWidget()
        self.acc_point_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.acc_point_list.itemDoubleClicked.connect(self.on_accuracy_point_list_double_clicked)
        self.acc_point_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.acc_point_list.customContextMenuRequested.connect(self.show_accuracy_point_list_context_menu)
        assign_layout.addWidget(self.acc_point_list)

        nav_row = QHBoxLayout()
        nav_row.addStretch()
        prev_btn = QPushButton("<")
        prev_btn.setMaximumWidth(36)
        prev_btn.setToolTip("Previous point (,)")
        prev_btn.clicked.connect(self.accuracy_nav_prev)
        self.acc_nav_label = QLabel("No points")
        next_btn = QPushButton(">")
        next_btn.setMaximumWidth(36)
        next_btn.setToolTip("Next point (.)")
        next_btn.clicked.connect(self.accuracy_nav_next)
        nav_row.addWidget(prev_btn)
        nav_row.addWidget(self.acc_nav_label)
        nav_row.addWidget(next_btn)
        nav_row.addStretch()
        assign_layout.addLayout(nav_row)

        assign_dd_row = QHBoxLayout()
        assign_dd_row.addWidget(QLabel("Assign checked to:"))
        self.acc_assign_class_combo = QComboBox()
        self.acc_assign_class_combo.currentTextChanged.connect(self.on_accuracy_assign_dropdown_changed)
        assign_dd_row.addWidget(self.acc_assign_class_combo, 1)
        assign_layout.addLayout(assign_dd_row)

        unassign_checked_btn = QPushButton("Unassign Checked")
        unassign_checked_btn.clicked.connect(self.unassign_checked_accuracy_points)
        assign_layout.addWidget(unassign_checked_btn)

        assign_group.setLayout(assign_layout)
        col2_layout.addWidget(assign_group, 1)   # grows to fill the column

        # --- Run Accuracy Assessment: this tab's own "final step", styled like
        # Export Final Classified TIF in Tab 3. ---
        self.acc_warn_label = QLabel("")
        self.acc_warn_label.setWordWrap(True)
        self.acc_warn_label.setStyleSheet("color: #a1531f; font-size: 10.5px;")
        col2_layout.addWidget(self.acc_warn_label)

        self.acc_run_btn = QPushButton("Run Accuracy Assessment")
        self.acc_run_btn.setMinimumHeight(48)
        self.acc_run_btn.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: white; font-weight: 600; border-radius: 4px; }"
            "QPushButton:hover { background-color: #a93226; }"
            "QPushButton:disabled { background-color: #d3a29c; color: #f2e6e4; }"
        )
        self.acc_run_btn.clicked.connect(self.run_accuracy_assessment)
        col2_layout.addWidget(self.acc_run_btn)

        reset_btn = QPushButton("Reset and Return to Loading")
        reset_btn.setMinimumHeight(34)
        reset_btn.setToolTip("Clear everything in memory and start again from Tab 1 (files on disk are kept)")
        reset_btn.clicked.connect(self.reset_interface)
        col2_layout.addWidget(reset_btn)

        # --- Viewer: raw ortho, no classification overlay, while labeling ---
        viewer_container = QWidget()
        viewer_layout = QVBoxLayout(viewer_container)
        viewer_layout.setContentsMargins(0, 0, 0, 0)

        overlay_bar = QHBoxLayout()
        self.acc_view_label = QLabel("View: raw orthomosaic (classification hidden while labelling)")
        overlay_bar.addWidget(self.acc_view_label)
        overlay_bar.addWidget(self._new_map_info(), 1)
        viewer_layout.addWidget(self._map_header([overlay_bar]))

        self.acc_viewer = InteractivePhotoViewer()
        self.acc_viewer.set_detail_provider(self._detail_image)
        self.acc_viewer.set_interaction_mode("points")
        self.acc_viewer.pointToggled.connect(self.on_accuracy_point_toggled)
        self.acc_viewer.pointContextMenuRequested.connect(self.on_accuracy_point_context_menu)
        self.acc_viewer.pointMoveStarted.connect(self.on_accuracy_point_move_started)
        self.acc_viewer.pointMoveDragged.connect(self.on_accuracy_point_move_dragged)
        self.acc_viewer.pointMoveFinished.connect(self.on_accuracy_point_move_finished)
        viewer_layout.addWidget(self.acc_viewer, 1)
        viewer_layout.addWidget(self._map_tip("Click a point to tick it; right-click to label it; Ctrl+drag to move it; "
                                              ", and . step through the points."))

        layout.addWidget(col1_widget)
        layout.addWidget(col2_widget)
        layout.addWidget(viewer_container)
        self.tabs.addTab(tab_acc, "4. Accuracy Assessment")
        self.accuracy_tab = tab_acc

        # ,/. as quick prev/next, mirroring </> without requiring the mouse -- scoped
        # to this tab (and everything inside it) so typing a literal ',' or '.' into an
        # unrelated field elsewhere in the app is never intercepted.
        prev_shortcut = QShortcut(QKeySequence(","), tab_acc)
        prev_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        prev_shortcut.activated.connect(self.accuracy_nav_prev)
        next_shortcut = QShortcut(QKeySequence("."), tab_acc)
        next_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        next_shortcut.activated.connect(self.accuracy_nav_next)

        self.sync_accuracy_gt_controls()
        self.refresh_iteration_table()

    def sync_accuracy_gt_controls(self):
        """ Mirrors the mockup's radio/checkbox sync: CSV path + "also generate
        expert-visual points" only make sense when the CSV path is selected, and the
        Random Point Generation group only makes sense when expert-visual points are
        actually going to be produced (either the CSV-free path, or CSV + supplement
        checked). Classes tally / Point Assignment / Run stay visible regardless --
        CSV-only points still benefit from being reviewable/editable there. """
        is_csv = self.acc_gt_csv_radio.isChecked()
        self.acc_csv_path_row_widget.setVisible(is_csv)
        self.acc_supp_expert_check.setVisible(is_csv)
        self.acc_csv_summary_label.setVisible(is_csv and bool(self.acc_csv_path_edit.text()))

        show_random_gen = self.acc_gt_expert_radio.isChecked() or (is_csv and self.acc_supp_expert_check.isChecked())
        self.acc_gen_group.setVisible(show_random_gen)

    def browse_for_accuracy_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Ground Truth CSV", "", "CSV Files (*.csv)")
        if not file_path:
            return
        if self.engine.ortho_data is None:
            print("Error: Load an orthomosaic first before importing ground-truth points.")
            return
        if not self.classes:
            print("Error: Define at least one class in Tab 3 before importing ground-truth points.")
            return
        try:
            n = self.engine.import_accuracy_points_csv(file_path, list(self.classes.keys()))
            self.acc_csv_path_edit.setText(file_path)
            trailing = " — no expert labeling needed for this run" if not self.acc_supp_expert_check.isChecked() else ""
            self.acc_csv_summary_label.setText(
                f"{n} points loaded from CSV.\nAll class names matched your Tab 3 class list.\nReady to run{trailing}."
            )
            print(f"Imported {n} ground-truth point(s) from CSV.")
            self.refresh_accuracy_ui()
        except Exception as e:
            QMessageBox.critical(self, "CSV Import Failed", str(e))
        self.sync_accuracy_gt_controls()

    def generate_accuracy_points(self):
        if self.engine.ortho_data is None:
            print("Error: Load an orthomosaic first before generating accuracy points.")
            return
        exclude_ids = set()
        for data in self.classes.values():
            exclude_ids |= data["training_segments"]
        sample_size = self.acc_sample_size_spinbox.value()
        try:
            if self.acc_design_combo.currentIndex() == 1:
                if self.sieved_classified_raster_ints is None:
                    print("Error: Classify first -- a stratified sample uses the final classified map's classes as strata.")
                    return
                self.engine.generate_stratified_accuracy_points(
                    self.sieved_classified_raster_ints, list(self.classes.keys()), sample_size,
                    self.acc_min_per_class_spinbox.value(), exclude_segment_ids=exclude_ids)
            else:
                self.engine.generate_random_accuracy_points(sample_size, exclude_segment_ids=exclude_ids)
            self.refresh_accuracy_ui()
        except Exception as e:
            print(f"Point generation failed: {e}")

    def refresh_accuracy_ui(self):
        """ Central refresh -- called after essentially any accuracy-point mutation
        (generate, import, assign, delete, move) or any class-management change that
        could affect the tally/colors/dropdowns. """
        self.update_accuracy_classes_table()
        self.update_accuracy_assign_combo()
        self.update_accuracy_point_list()
        self.update_accuracy_nav_label()
        self.update_accuracy_warn_label()
        self.update_accuracy_map_display()

    def update_accuracy_classes_table(self):
        gdf = self.engine.accuracy_points_gdf
        counts = {}
        if gdf is not None and len(gdf) > 0:
            counts = gdf['reference_class'].value_counts().to_dict()
        self.acc_classes_table.blockSignals(True)
        self.acc_classes_table.setRowCount(len(self.classes))
        for row, (name, data) in enumerate(self.classes.items()):
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            name_item.setBackground(data["color"])
            name_item.setForeground(QColor(0, 0, 0))
            count_item = QTableWidgetItem(str(int(counts.get(name, 0))))
            count_item.setFlags(count_item.flags() & ~Qt.ItemIsEditable)
            count_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.acc_classes_table.setItem(row, 0, name_item)
            self.acc_classes_table.setItem(row, 1, count_item)
        self.acc_classes_table.blockSignals(False)

    def on_accuracy_classes_table_selection_changed(self):
        selected = self.acc_classes_table.selectedItems()
        if not selected:
            return
        name_item = self.acc_classes_table.item(selected[0].row(), 0)
        if name_item:
            idx = self.acc_filter_combo.findText(name_item.text())
            if idx >= 0:
                self.acc_filter_combo.setCurrentIndex(idx)

    def update_accuracy_assign_combo(self):
        self.acc_assign_class_combo.blockSignals(True)
        current = self.acc_assign_class_combo.currentText()
        self.acc_assign_class_combo.clear()
        self.acc_assign_class_combo.addItems(list(self.classes.keys()))
        idx = self.acc_assign_class_combo.findText(current)
        self.acc_assign_class_combo.setCurrentIndex(idx if idx >= 0 else (0 if self.classes else -1))
        self.acc_assign_class_combo.blockSignals(False)

        self.acc_filter_combo.blockSignals(True)
        current_filter = self.acc_filter_combo.currentText()
        self.acc_filter_combo.clear()
        self.acc_filter_combo.addItem("All")
        self.acc_filter_combo.addItem("Unassigned")
        self.acc_filter_combo.addItems(list(self.classes.keys()))
        idx = self.acc_filter_combo.findText(current_filter)
        self.acc_filter_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.acc_filter_combo.blockSignals(False)

    def get_filtered_accuracy_points(self):
        """ point_ids matching the current filter dropdown, in table order. """
        gdf = self.engine.accuracy_points_gdf
        if gdf is None or len(gdf) == 0:
            return []
        filt = self.acc_filter_combo.currentText() if hasattr(self, 'acc_filter_combo') else "All"
        if not filt or filt == "All":
            rows = gdf
        elif filt == "Unassigned":
            rows = gdf[gdf['reference_class'].isna()]
        else:
            rows = gdf[gdf['reference_class'] == filt]
        return [int(p) for p in rows['point_id']]

    def update_accuracy_point_list(self):
        # Rebuilding wipes the scrollbar position and, since every row's checkbox/chip
        # is a real QWidget that gets destroyed and recreated, can silently drop
        # keyboard focus too (which breaks the ,/. shortcuts -- their
        # WidgetWithChildrenShortcut context needs *something* inside the tab to still
        # have focus). Capture both before clearing and restore them after.
        scroll_val = self.acc_point_list.verticalScrollBar().value()
        self.acc_point_list.blockSignals(True)
        self.acc_point_list.clear()
        gdf = self.engine.accuracy_points_gdf
        filtered_ids = self.get_filtered_accuracy_points()
        by_id = {int(r.point_id): r for r in gdf.itertuples(index=False)} if gdf is not None else {}
        class_colors = {name: data["color"] for name, data in self.classes.items()}

        for pid in filtered_ids:
            row = by_id.get(pid)
            if row is None:
                continue
            item = QListWidgetItem(self.acc_point_list)
            item.setData(Qt.UserRole, pid)
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(4, 2, 4, 2)

            checkbox = QCheckBox()
            checkbox.setChecked(pid in self.acc_checked_point_ids)
            checkbox.toggled.connect(lambda checked, p=pid: self.on_accuracy_point_checkbox_toggled(p, checked))
            row_layout.addWidget(checkbox)

            geom = row.geometry
            label = QLabel(f"Pt #{pid:03d}  ({geom.x:.1f}, {geom.y:.1f})")
            row_layout.addWidget(label, 1)

            ref_class = row.reference_class
            ref_class = None if pd.isna(ref_class) else ref_class
            chip = QPushButton(ref_class if ref_class else "Unassigned")
            chip.setCursor(Qt.PointingHandCursor)
            chip.setToolTip("Click to change this point's class assignment")
            if ref_class and ref_class in class_colors:
                c = class_colors[ref_class]
                chip.setStyleSheet(
                    f"QPushButton {{ background-color: {c.name()}; color: black; border: none; "
                    "border-radius: 8px; padding: 1px 6px; font-size: 10px; }"
                )
            else:
                chip.setStyleSheet(
                    "QPushButton { background-color: #999; color: #eee; border: none; "
                    "border-radius: 8px; padding: 1px 6px; font-size: 10px; }"
                )
            chip.clicked.connect(lambda checked=False, p=pid, w=chip: self.show_accuracy_point_chip_menu(p, w))
            row_layout.addWidget(chip)

            item.setSizeHint(row_widget.sizeHint())
            self.acc_point_list.setItemWidget(item, row_widget)
        self.acc_point_list.blockSignals(False)
        self.acc_point_list.verticalScrollBar().setValue(scroll_val)
        # Only steal focus back while this tab is actually the one on screen -- a
        # refresh triggered as a side effect of Tab 2 class management (rename/merge/
        # add/remove all call refresh_accuracy_ui) must never yank focus away from
        # wherever the user actually is.
        if self.tabs.currentWidget() is getattr(self, 'acc_tab_widget', None):
            self.acc_point_list.setFocus(Qt.OtherFocusReason)

    def on_accuracy_point_list_double_clicked(self, item):
        """ Double-click a row (its label, not the checkbox or class chip -- those
        consume the click themselves) to pan/zoom the map straight to that point,
        without needing to check it and step there one at a time via </> or ,/. . """
        point_id = item.data(Qt.UserRole)
        if point_id is None:
            return
        filtered = self.get_filtered_accuracy_points()
        if point_id in filtered:
            self.acc_nav_index = filtered.index(point_id)
            self.update_accuracy_nav_label()
        self.center_accuracy_viewer_on_point(point_id)

    def show_accuracy_point_chip_menu(self, point_id, anchor_widget):
        """ Popup menu anchored under a point-list row's class chip, letting that one
        point's assignment be changed (or cleared) directly from the list -- the
        list-side counterpart to the map's right-click "Assign"/"Unassign" menu. """
        menu = QMenu(self)
        for class_name, data in self.classes.items():
            action = menu.addAction(class_name)
            swatch = QPixmap(10, 10)
            swatch.fill(data["color"])
            action.setIcon(QIcon(swatch))
            action.triggered.connect(lambda checked=False, p=point_id, c=class_name: self.assign_accuracy_points([p], c))
        menu.addSeparator()
        unassign_action = menu.addAction("Unassign")
        unassign_action.triggered.connect(lambda: self.unassign_accuracy_points([point_id]))
        menu.exec(anchor_widget.mapToGlobal(anchor_widget.rect().bottomLeft()))

    def on_accuracy_point_checkbox_toggled(self, point_id, checked):
        if checked:
            self.acc_checked_point_ids.add(point_id)
        else:
            self.acc_checked_point_ids.discard(point_id)
        self.update_accuracy_map_display()

    def on_accuracy_select_all_toggled(self, checked):
        filtered_ids = set(self.get_filtered_accuracy_points())
        if checked:
            self.acc_checked_point_ids |= filtered_ids
        else:
            self.acc_checked_point_ids -= filtered_ids
        self.update_accuracy_point_list()
        self.update_accuracy_map_display()

    def on_accuracy_filter_changed(self, text):
        self.acc_nav_index = -1
        self.update_accuracy_point_list()
        self.update_accuracy_nav_label()

    def update_accuracy_nav_label(self):
        filtered = self.get_filtered_accuracy_points()
        if not filtered:
            self.acc_nav_label.setText("No points")
            return
        idx = self.acc_nav_index if 0 <= self.acc_nav_index < len(filtered) else 0
        self.acc_nav_label.setText(f"Point {idx + 1} of {len(filtered)} (filtered)")

    def accuracy_nav_prev(self):
        self._accuracy_nav_step(-1)

    def accuracy_nav_next(self):
        self._accuracy_nav_step(1)

    def _accuracy_nav_step(self, delta):
        filtered = self.get_filtered_accuracy_points()
        if not filtered:
            return
        self.acc_nav_index = (self.acc_nav_index + delta) % len(filtered)
        point_id = filtered[self.acc_nav_index]
        self.update_accuracy_nav_label()
        self.center_accuracy_viewer_on_point(point_id)
        self.focus_accuracy_list_row(point_id)

    def focus_accuracy_list_row(self, point_id):
        """ Selects and scrolls the Point Assignment list to the row for point_id --
        keeps the list in sync with </> and ,/. navigation, mirroring what already
        happens on the map. """
        for i in range(self.acc_point_list.count()):
            item = self.acc_point_list.item(i)
            if item.data(Qt.UserRole) == point_id:
                self.acc_point_list.setCurrentItem(item)
                self.acc_point_list.scrollToItem(item)
                break

    def accuracy_point_pixel_pos(self, point_id):
        gdf = self.engine.accuracy_points_gdf
        if gdf is None or self.engine.ortho_meta is None:
            return None
        rows = gdf[gdf['point_id'] == point_id]
        if len(rows) == 0:
            return None
        geom = rows.iloc[0].geometry
        transform = self.engine.ortho_meta['transform']
        col, row = ~transform * (geom.x, geom.y)
        return col, row

    def center_accuracy_viewer_on_point(self, point_id):
        pixel = self.accuracy_point_pixel_pos(point_id)
        if pixel is None:
            return
        x, y = pixel
        # 3.2 = one wheel-zoom-out step below the previous 4.0 (the viewer's own
        # zoom-in factor is 1.25/step, so 4.0 / 1.25 = 3.2) -- 4.0 was judged too close.
        self.acc_viewer.center_on_pixmap_point(x, y, zoom=3.2)

    def find_nearest_accuracy_point(self, pos, max_dist_px=None):
        """ Nearest point within about 10 screen pixels (markers are drawn at a fixed
        on-screen size, so the click tolerance follows the zoom). """
        gdf = self.engine.accuracy_points_gdf
        if gdf is None or len(gdf) == 0 or self.engine.ortho_meta is None:
            return None
        if max_dist_px is None:
            max_dist_px = 10.0 / max(self.acc_viewer.transform().m11(), 1e-6)
        transform = self.engine.ortho_meta['transform']
        best_id, best_dist = None, max_dist_px
        for row in gdf.itertuples(index=False):
            col, r = ~transform * (row.geometry.x, row.geometry.y)
            dist = ((col - pos.x()) ** 2 + (r - pos.y()) ** 2) ** 0.5
            if dist <= best_dist:
                best_dist = dist
                best_id = int(row.point_id)
        return best_id

    def on_accuracy_point_toggled(self, pos):
        point_id = self.find_nearest_accuracy_point(pos)
        if point_id is None:
            return
        if point_id in self.acc_checked_point_ids:
            self.acc_checked_point_ids.discard(point_id)
        else:
            self.acc_checked_point_ids.add(point_id)
        self.update_accuracy_point_list()
        self.update_accuracy_map_display()

    def on_accuracy_point_context_menu(self, pixmap_pos, global_pos):
        point_id = self.find_nearest_accuracy_point(pixmap_pos)
        if point_id is None:
            self._show_accuracy_add_point_menu(pixmap_pos, global_pos)
            return
        menu = QMenu(self)
        for class_name, data in self.classes.items():
            action = menu.addAction(class_name)
            swatch = QPixmap(10, 10)
            swatch.fill(data["color"])
            action.setIcon(QIcon(swatch))
            action.triggered.connect(lambda checked=False, p=point_id, c=class_name: self.assign_accuracy_points([p], c))
        unassign_action = menu.addAction("Unassign")
        unassign_action.triggered.connect(lambda: self.unassign_accuracy_points([point_id]))
        menu.addSeparator()
        move_hint = menu.addAction("Move Point (Ctrl+drag)")
        move_hint.setEnabled(False)
        delete_action = menu.addAction("Delete Point")
        delete_action.triggered.connect(lambda: self.delete_accuracy_points([point_id]))
        menu.exec(global_pos)

    def _show_accuracy_add_point_menu(self, pixmap_pos, global_pos):
        """ Right-click on empty map space (no existing point nearby) -- lets the user
        drop a brand-new point right there and, in the same gesture, either assign it
        a class immediately or leave it unassigned for later. This is the primary way
        to manually add accuracy points (there's no separate "add point" mode/button --
        the map itself is the entry point, consistent with how Tab 2's training-sample
        workflow keeps everything on the map). """
        if self.engine.ortho_meta is None:
            return
        menu = QMenu(self)
        for class_name, data in self.classes.items():
            action = menu.addAction(f"Add Point: {class_name}")
            swatch = QPixmap(10, 10)
            swatch.fill(data["color"])
            action.setIcon(QIcon(swatch))
            action.triggered.connect(
                lambda checked=False, pos=pixmap_pos, c=class_name: self.add_manual_accuracy_point(pos, c)
            )
        if self.classes:
            menu.addSeparator()
        unassigned_action = menu.addAction("Add Point (Unassigned)")
        unassigned_action.triggered.connect(
            lambda checked=False, pos=pixmap_pos: self.add_manual_accuracy_point(pos, None)
        )
        menu.exec(global_pos)

    def add_manual_accuracy_point(self, pixmap_pos, class_name):
        """ App-level wrapper for the map's "Add Point" gesture -- converts the
        clicked pixel position to real-world (x, y) via the ortho's affine transform,
        then delegates to the engine. """
        if self.engine.ortho_meta is None:
            return
        transform = self.engine.ortho_meta['transform']
        x, y = transform * (pixmap_pos.x(), pixmap_pos.y())
        point_id = self.engine.add_manual_accuracy_point(x, y, reference_class=class_name)
        label = class_name if class_name else "unassigned"
        print(f"Added point #{point_id} ({label}).")
        self.refresh_accuracy_ui()

    def assign_accuracy_points(self, point_ids, class_name):
        if not point_ids or self.engine.accuracy_points_gdf is None:
            return
        self.engine.set_accuracy_point_class(point_ids, class_name)
        print(f"Assigned {len(point_ids)} point(s) to class '{class_name}'.")
        self.refresh_accuracy_ui()

    def delete_accuracy_points(self, point_ids):
        if not point_ids or self.engine.accuracy_points_gdf is None:
            return
        self.engine.delete_accuracy_points(point_ids)
        self.acc_checked_point_ids -= set(point_ids)
        print(f"Deleted {len(point_ids)} point(s).")
        self.refresh_accuracy_ui()

    def show_accuracy_point_list_context_menu(self, pos):
        """ Right-click on the Point Assignment list -- operates on the list's own
        (shift/ctrl-extendable) selection, not the checkbox "checked" set used by
        Assign/Unassign. Right-clicking a row outside the current selection selects
        just that row first, matching standard list/file-manager behavior. """
        item = self.acc_point_list.itemAt(pos)
        if item is not None and item not in self.acc_point_list.selectedItems():
            self.acc_point_list.setCurrentItem(item)
        point_ids = [it.data(Qt.UserRole) for it in self.acc_point_list.selectedItems()]
        if not point_ids:
            return
        menu = QMenu(self)
        label = f"Delete {len(point_ids)} Point(s)" if len(point_ids) > 1 else "Delete Point"
        delete_action = menu.addAction(label)
        delete_action.triggered.connect(lambda: self._confirm_delete_accuracy_points(point_ids))
        menu.exec(self.acc_point_list.viewport().mapToGlobal(pos))

    def _confirm_delete_accuracy_points(self, point_ids):
        if len(point_ids) > 1:
            reply = QMessageBox.question(
                self, "Delete Points",
                f"Delete {len(point_ids)} selected point(s)? This cannot be undone.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        self.delete_accuracy_points(point_ids)

    def unassign_accuracy_points(self, point_ids):
        """ Clears reference_class back to None/unassigned for the given points --
        distinct from Delete, which removes the point entirely. Used by both the map's
        right-click context menu (single point) and the Point Assignment panel's
        "Unassign Checked" button (multiple). """
        if not point_ids or self.engine.accuracy_points_gdf is None:
            return
        self.engine.set_accuracy_point_class(point_ids, None)
        print(f"Unassigned {len(point_ids)} point(s).")
        self.refresh_accuracy_ui()

    def unassign_checked_accuracy_points(self):
        if not self.acc_checked_point_ids:
            print("No points are checked.")
            return
        self.unassign_accuracy_points(list(self.acc_checked_point_ids))

    def clear_accuracy_selection(self):
        """ Deselects (unchecks) every currently multi-selected point, without
        touching their class assignments -- the counterpart to "Select all" that was
        previously missing. """
        if not self.acc_checked_point_ids:
            return
        self.acc_checked_point_ids.clear()
        if hasattr(self, 'acc_select_all_check'):
            self.acc_select_all_check.blockSignals(True)
            self.acc_select_all_check.setChecked(False)
            self.acc_select_all_check.blockSignals(False)
        self.update_accuracy_point_list()
        self.update_accuracy_map_display()

    def _ensure_workspace_dir(self):
        """ Creates self.workspace_dir on disk the first time something is actually
        saved into it -- deliberately not done just because an ortho was loaded or
        browsed to (see load_ortho), so relocating a moved raster during Load
        Workspace can't leave a stray empty "<name>_workspace" folder behind next to
        wherever that file happens to now live. """
        if self.workspace_dir:
            os.makedirs(self.workspace_dir, exist_ok=True)

    def save_accuracy_progress(self):
        """ Lightweight manual save for accuracy points only -- writes
        accuracy_points.gpkg (and, if a session.json already exists, refreshes its
        accuracy-related keys) without redoing the full Save Workspace Session
        (segments/features/classified map). Points are otherwise in-memory only and
        would be lost on a crash; this button and Save Workspace Session (Tab 1) are
        the only two things that persist them -- Load Workspace Session restores
        them automatically either way. """
        if self.workspace_dir is None:
            print("Error: Load an orthomosaic first.")
            return
        gdf = self.engine.accuracy_points_gdf
        if gdf is None or len(gdf) == 0:
            print("No accuracy points to save yet.")
            return
        try:
            self._ensure_workspace_dir()
            gdf.to_file(os.path.join(self.workspace_dir, "accuracy_points.gpkg"), driver="GPKG")
            session_json = os.path.join(self.workspace_dir, "session.json")
            if os.path.exists(session_json):
                with open(session_json, 'r') as f:
                    session_data = json.load(f)
                session_data["accuracy_sample_size"] = self.acc_sample_size_spinbox.value()
                session_data["accuracy_strata"] = self.engine.accuracy_strata
                session_data["accuracy_next_point_id"] = self.engine.next_accuracy_point_id
                with open(session_json, 'w') as f:
                    json.dump(session_data, f, indent=4)
            else:
                print(
                    "Note: accuracy_points.gpkg was saved, but no session.json exists yet -- use "
                    "File > Save Workspace (Ctrl+S) at least once so File > Load Workspace can "
                    "find it."
                )
            print(f"Accuracy progress saved ({len(gdf)} point(s)).")
        except Exception as e:
            print(f"Save failed: {e}")

    def on_accuracy_assign_dropdown_changed(self, class_name):
        """ Auto-apply on selection, consistent with the rest of the app's immediate-
        feedback style -- no separate "Apply" click needed. No-op if nothing is
        currently checked, matching the plan's "applies to every checked point above". """
        if not class_name or not self.acc_checked_point_ids:
            return
        self.assign_accuracy_points(list(self.acc_checked_point_ids), class_name)

    def on_accuracy_point_move_started(self, pos):
        self.acc_point_move_id = self.find_nearest_accuracy_point(pos)

    def on_accuracy_point_move_dragged(self, pos):
        if self.acc_point_move_id is None or self.engine.ortho_meta is None:
            return
        transform = self.engine.ortho_meta['transform']
        x, y = transform * (pos.x(), pos.y())
        self.engine.move_accuracy_point(self.acc_point_move_id, x, y)
        self.update_accuracy_map_display()

    def on_accuracy_point_move_finished(self, pos):
        if self.acc_point_move_id is not None:
            self.on_accuracy_point_move_dragged(pos)
            print(f"Moved point #{self.acc_point_move_id}.")
        self.acc_point_move_id = None
        self.update_accuracy_point_list()

    def update_accuracy_warn_label(self):
        gdf = self.engine.accuracy_points_gdf
        if gdf is None or len(gdf) == 0:
            self.acc_warn_label.setText("")
            return
        total = len(gdf)
        unassigned = int(gdf['reference_class'].isna().sum())
        if unassigned > 0:
            self.acc_warn_label.setText(f"{unassigned} of {total} points are still unassigned — they'll be excluded from this run.")
        else:
            self.acc_warn_label.setText("")

    def update_accuracy_map_display(self):
        if not hasattr(self, 'acc_viewer') or self.raw_pixmap is None:
            return
        self.acc_viewer.set_base_image_cached(self.raw_pixmap, grid_size=self.raw_grid)
        self.draw_accuracy_point_markers()

    def draw_accuracy_point_markers(self, qimg=None):
        """ Accuracy points as vector markers over the map (radius 6 px, 70 % opacity), so
        the land under them stays visible and nothing full-size is redrawn per click. """
        gdf = self.engine.accuracy_points_gdf
        if gdf is None or len(gdf) == 0 or self.engine.ortho_meta is None:
            self.acc_viewer.set_markers([])
            return
        transform = self.engine.ortho_meta['transform']
        class_colors = {name: data["color"] for name, data in self.classes.items()}
        markers = []
        for row in gdf.itertuples(index=False):
            col, r = ~transform * (row.geometry.x, row.geometry.y)
            ref_class = row.reference_class
            ref_class = None if pd.isna(ref_class) else ref_class
            if ref_class and ref_class in class_colors:
                fill, pen = class_colors[ref_class], QColor(255, 255, 255)
            else:
                fill, pen = QColor(255, 255, 255), QColor(0, 0, 0)
            markers.append((float(col), float(r), 6.0, fill, pen, int(row.point_id) in self.acc_checked_point_ids))
        self.acc_viewer.set_markers(markers, opacity=0.7)

    def run_accuracy_assessment(self):
        gdf = self.engine.accuracy_points_gdf
        if gdf is None or len(gdf) == 0:
            print("Error: No accuracy points exist yet -- generate or import some first.")
            return
        if not self.classes:
            print("Error: Define classes in Tab 3 first.")
            return
        if self.sieved_classified_raster_ints is None:
            print("Error: Run classification (Tab 3) first -- accuracy assessment reads the final classified map.")
            return
        if self.workspace_dir is None:
            print("Error: Load an orthomosaic first (no active workspace).")
            return

        total = len(gdf)
        unassigned = int(gdf['reference_class'].isna().sum())
        if unassigned > 0:
            reply = QMessageBox.question(
                self, "Run Accuracy Assessment",
                f"{unassigned} of {total} points are still unassigned and will be excluded from this run. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        try:
            print("Running accuracy assessment against the final classified map...")
            QApplication.processEvents()
            class_names_ordered = list(self.classes.keys())
            result = self.engine.run_accuracy_assessment(
                self.sieved_classified_raster_ints, class_names_ordered, self.workspace_dir
            )
            print(f"Accuracy assessment complete: Overall Accuracy = {result['overall_accuracy'] * 100:.1f}% "
                  f"({result['n_labeled']} labeled points used).")
            st = result.get('stratified')
            if st is not None:
                print(f" -> Stratified sample, area-weighted: Overall Accuracy = {st['oa'] * 100:.1f}% "
                      f"± {1.96 * st['oa_se'] * 100:.1f}% (95% CI, {st['n_used']} points).")
            print(f" -> Wrote {result['csv_path']}")
            print(f" -> Wrote {result['gpkg_path']}")
            self.refresh_accuracy_ui()
        except Exception as e:
            print(f"Accuracy assessment failed: {e}")
            return

        class_colors = {name: data["color"] for name, data in self.classes.items()}
        image = render_confusion_matrix_image(result, class_colors)
        png_path = os.path.join(self.workspace_dir, "accuracy_matrix.png")
        if image.save(png_path):
            print(f" -> Wrote {png_path}")
        else:
            print(f"Warning: could not write {png_path}")
        it = self._record_iteration(result, image)
        self.show_accuracy_matrix_dialog(image, f"Saved to: {png_path}", f"Iteration {it['n']}")

    def show_accuracy_matrix_dialog(self, image, note, title_suffix=None):
        """ Shows the styled confusion-matrix figure (also saved as accuracy_matrix.png
        in the workspace). The image is rendered at 2x, so it's shown at half size.
        Non-modal: the main window stays usable (e.g. to inspect points) while it's open;
        a new run replaces the previous results window. """
        old = getattr(self, "_acc_dialog", None)
        if old is not None:
            try:
                old.close()
            except RuntimeError:
                pass
        dialog = QDialog(self)
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        dialog.setModal(False)
        dialog.setWindowTitle("Accuracy Assessment Results" + (f" — {title_suffix}" if title_suffix else ""))
        self._acc_dialog = dialog
        # only forget it if it is still the current one (a replaced window closes later)
        dialog.finished.connect(lambda *_, d=dialog: self._acc_dialog is d and setattr(self, "_acc_dialog", None))
        layout = QVBoxLayout(dialog)

        pixmap = QPixmap.fromImage(image)
        pixmap.setDevicePixelRatio(2)
        label = QLabel()
        label.setPixmap(pixmap)
        scroll = QScrollArea()
        scroll.setWidget(label)
        scroll.setAlignment(Qt.AlignCenter)
        logical_w, logical_h = image.width() // 2, image.height() // 2
        screen = self.screen().availableGeometry()
        scroll.setMinimumSize(min(logical_w + 4, screen.width() - 120), min(logical_h + 4, screen.height() - 200))
        layout.addWidget(scroll)

        path_label = QLabel(note)
        path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        path_label.setStyleSheet("color: #76817b;")
        layout.addWidget(path_label)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        copy_btn = QPushButton("Copy Image")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setImage(image))
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        buttons.addWidget(copy_btn)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

        dialog.show()

    # --- Iterations: snapshot at each accuracy run, restore to roll back ---
    _LUT_ROWS = 512   # row block for the segment <-> class-map conversions (keeps index temporaries small)

    def _class_map_lut(self):
        """ The raw classified map as one class index per segment id (classify_all assigns
        whole segments, so this loses nothing and is tiny next to the raster). """
        raw, seg = self.raw_classified_raster_ints, self.engine.segmented_image
        if raw is None or seg is None or raw.shape != seg.shape:
            return None
        lut = np.zeros(int(seg.max()) + 1, dtype=np.uint8)
        for r in range(0, seg.shape[0], self._LUT_ROWS):
            lut[seg[r:r + self._LUT_ROWS]] = raw[r:r + self._LUT_ROWS]
        return lut

    def _class_map_from_lut(self, lut):
        seg = self.engine.segmented_image
        top = int(seg.max())
        if top >= len(lut):
            lut = np.concatenate([lut, np.zeros(top + 1 - len(lut), dtype=np.uint8)])
        out = np.empty(seg.shape, dtype=np.uint8)
        for r in range(0, seg.shape[0], self._LUT_ROWS):
            out[r:r + self._LUT_ROWS] = lut[seg[r:r + self._LUT_ROWS]]
        return out

    @staticmethod
    def _iteration_key(classes, groups, points):
        """ Stable fingerprint of what a restore replaces (samples, names, points), to
        tell whether the current state is already kept in the list. """
        import hashlib
        pts = None
        if points is not None and len(points):
            pts = sorted(zip(points["point_id"].astype(int).tolist(), points["reference_class"].astype(str).tolist(),
                             points.geometry.x.round(3).tolist(), points.geometry.y.round(3).tolist()))
        grp = None if groups is None else {k: [v.get("name"), v.get("status")] for k, v in groups["groups"].items()}
        blob = json.dumps([classes, grp, pts], sort_keys=True, default=str)
        return hashlib.md5(blob.encode()).hexdigest()

    def _current_iteration_state(self):
        classes = [[n, d["color"].name(), sorted(int(x) for x in d["training_segments"])]
                   for n, d in self.classes.items()]
        groups = None if self.groups is None else json.loads(json.dumps(self.groups.to_json()))
        gdf = self.engine.accuracy_points_gdf
        points = None if gdf is None or len(gdf) == 0 else gdf.copy()
        return classes, groups, points

    def _record_iteration(self, result=None, image=None):
        """ Snapshots the state an assessment was run on (or, with no result, the current
        state as an unassessed checkpoint) and adds it to the Iterations list. """
        classes, groups, points = self._current_iteration_state()
        st = (result or {}).get("stratified")
        it = {
            "n": max([i["n"] for i in self.iterations], default=0) + 1,
            "time": time.strftime("%Y-%m-%d %H:%M"),
            "oa": None if result is None else float(result["overall_accuracy"]),
            "n_labeled": None if result is None else int(result["n_labeled"]),
            "oa_strat": None if st is None else float(st["oa"]),
            "oa_ci": None if st is None else float(1.96 * st["oa_se"]),
            "classes": classes,
            "groups": groups,
            "points": points,
            "next_point_id": int(self.engine.next_accuracy_point_id),
            "strata": (json.loads(json.dumps(self.engine.accuracy_strata, default=float))
                       if self.engine.accuracy_strata else None),
            "lut": self._class_map_lut(),
            "sieve": int(self.gen_spinbox.value()),
            "use_deep": bool(self.use_deep_check.isChecked()),
            "seg_shape": list(self.engine.segmented_image.shape),
            "image": image,
        }
        it["key"] = self._iteration_key(classes, groups, points)
        self.iterations.append(it)
        self.refresh_iteration_table(select_n=it["n"])
        what = "Iteration" if result is not None else "Checkpoint"
        print(f" -> {what} {it['n']} kept in the Iterations list ({self._iteration_summary(it)}).")
        return it

    @staticmethod
    def _iteration_counts(it):
        n_samples = sum(len(c[2]) for c in it["classes"])
        labeled = 0 if it["points"] is None else int(it["points"]["reference_class"].notna().sum())
        return n_samples, labeled

    def _iteration_summary(self, it):
        n_samples, labeled = self._iteration_counts(it)
        parts = [f"{len(it['classes'])} classes", f"{n_samples} training samples", f"{labeled} labeled points"]
        if it["groups"] is not None:
            named = sum(1 for g in it["groups"]["groups"].values() if g.get("status") == "named")
            parts.append(f"{named} named groups")
        return ", ".join(parts)

    def refresh_iteration_table(self, select_n=None):
        if not hasattr(self, "iter_table"):
            return
        t = self.iter_table
        if select_n is None:
            cur = self._selected_iteration()
            select_n = cur["n"] if cur else None
        t.blockSignals(True)
        t.setRowCount(0)
        best = max((i["oa"] for i in self.iterations if i["oa"] is not None), default=None)
        for row, it in enumerate(self.iterations):
            t.insertRow(row)
            if it["oa"] is None:
                acc = "checkpoint"
            else:
                acc = f"{it['oa'] * 100:.1f}%"
                if it["oa_strat"] is not None:
                    acc += f"  (strat. {it['oa_strat'] * 100:.1f}%)"
            n_samples, labeled = self._iteration_counts(it)
            tip = f"Iteration {it['n']} — {it['time']}\n{self._iteration_summary(it)}"
            if it["oa_ci"] is not None:
                tip += f"\nStratified OA {it['oa_strat'] * 100:.1f}% ± {it['oa_ci'] * 100:.1f}% (95% CI)"
            if it["oa"] is None:
                tip += "\nKept before a restore; not assessed."
            for col, text in enumerate([str(it["n"]), acc, str(n_samples), str(labeled)]):
                item = QTableWidgetItem(text)
                item.setToolTip(tip)
                item.setData(Qt.UserRole, it["n"])
                if col != 1:
                    item.setTextAlignment(Qt.AlignCenter)
                if best is not None and it["oa"] == best:
                    f = item.font()
                    f.setBold(True)
                    item.setFont(f)
                if it["oa"] is None:
                    item.setForeground(QColor("#888888"))
                t.setItem(row, col, item)
            if it["n"] == select_n:
                t.selectRow(row)
        t.blockSignals(False)
        self._sync_iteration_buttons()

    def _selected_iteration(self):
        if not hasattr(self, "iter_table"):
            return None
        rows = self.iter_table.selectionModel().selectedRows()
        if not rows:
            return None
        n = self.iter_table.item(rows[0].row(), 0).data(Qt.UserRole)
        return next((i for i in self.iterations if i["n"] == n), None)

    def _sync_iteration_buttons(self):
        it = self._selected_iteration()
        self.iter_restore_btn.setEnabled(it is not None)
        self.iter_delete_btn.setEnabled(it is not None)
        self.iter_view_btn.setEnabled(it is not None and it["image"] is not None)

    def view_iteration_results(self):
        it = self._selected_iteration()
        if it is None or it["image"] is None:
            return
        self.show_accuracy_matrix_dialog(
            it["image"], f"Iteration {it['n']}, assessed {it['time']} ({self._iteration_summary(it)}).",
            f"Iteration {it['n']}")

    def delete_selected_iteration(self):
        it = self._selected_iteration()
        if it is None:
            return
        reply = QMessageBox.question(self, "Delete Iteration",
                                     f"Remove iteration {it['n']} from the list? It can't be restored afterwards.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        self.iterations.remove(it)
        self.refresh_iteration_table()
        print(f"Iteration {it['n']} removed from the list.")

    def restore_selected_iteration(self):
        it = self._selected_iteration()
        if it is None:
            return
        seg = self.engine.segmented_image
        if seg is None or self.engine.features_df is None or list(seg.shape) != list(it["seg_shape"]):
            print("Restore Error: this iteration belongs to a segmentation that is no longer loaded.")
            return

        current_kept = self._iteration_key(*self._current_iteration_state()) in {i["key"] for i in self.iterations}
        box = QMessageBox(self)
        box.setWindowTitle("Restore Iteration")
        box.setIcon(QMessageBox.Question)
        acc = f" (Overall Accuracy {it['oa'] * 100:.1f}%)" if it["oa"] is not None else ""
        box.setText(f"Return to iteration {it['n']}{acc}?")
        box.setInformativeText(
            "The classes and their training samples, Discover group names, accuracy points and classified "
            "map are replaced with the ones from that iteration, and the classifier is retrained on its "
            "samples. Other iterations stay in the list.")
        keep_check = QCheckBox("First keep the current state in the list as a checkpoint")
        keep_check.setChecked(not current_kept)
        box.setCheckBox(keep_check)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Yes)
        if box.exec() != QMessageBox.Yes:
            return
        if keep_check.isChecked():
            self._record_iteration()
        self._apply_iteration(it)

    def _apply_iteration(self, it):
        print(f"Restoring iteration {it['n']} ({it['time']})...")
        QApplication.processEvents()
        present = set(int(x) for x in self.engine.features_df.index)

        # classes and training samples (ids merged away since then are dropped)
        dropped = 0
        self.classes = {}
        self.class_list_widget.clear()
        for name, color_hex, ids in it["classes"]:
            keep = set(int(x) for x in ids) & present
            dropped += len(ids) - len(keep)
            color = QColor(color_hex)
            self.classes[name] = {"color": color, "training_segments": keep}
            item = QListWidgetItem(f"{name} ({len(keep)})")
            item.setBackground(color)
            self.class_list_widget.addItem(item)
        if dropped:
            print(f" -> {dropped} training sample(s) refer to segments merged since; they were left out.")
        self.selected_segments = set()
        self.locked_selection = set()
        self.current_seed_segments = set()
        self.selection_undo_stack = []
        self.selection_count_label.setText("Selected: 0 segments")
        self.rebuild_sample_class_map()

        # Discover group names
        if it["groups"] is None:
            print(" -> This iteration had no Discover groups.")
        self._discover_restore_groups(json.loads(json.dumps(it["groups"])) if it["groups"] else None,
                                      f"restored from iteration {it['n']}")

        # accuracy points (ids keep counting up, so a point added later never reuses an id)
        self.engine.accuracy_points_gdf = None if it["points"] is None else it["points"].copy()
        self.engine.next_accuracy_point_id = max(int(it["next_point_id"]), int(self.engine.next_accuracy_point_id))
        self.engine.accuracy_strata = json.loads(json.dumps(it["strata"])) if it["strata"] else None
        self.acc_checked_point_ids = set()
        self.acc_nav_index = -1

        # classifier settings, then the model (same samples -> same forest) and the map
        self.use_deep_check.setChecked(it["use_deep"])
        self.gen_spinbox.setValue(it["sieve"])
        self.classifier = None
        self.classifier_features = None
        if sum(1 for d in self.classes.values() if d["training_segments"]) >= 2:
            self.train_classifier()
        self._invalidate_display_caches()
        if it["lut"] is not None:
            self.raw_classified_raster_ints = self._class_map_from_lut(it["lut"])
            self.sieved_classified_raster_ints = self.raw_classified_raster_ints.copy()
            self.apply_generalization_sieve()
        else:
            self.raw_classified_raster_ints = None
            self.sieved_classified_raster_ints = None
            self.classified_image_data = None
            self.classified_qimg = None
            self.setup_classification_view()
        self.refresh_accuracy_ui()
        self.refresh_iteration_table(select_n=it["n"])
        print(f"Iteration {it['n']} restored: {self._iteration_summary(it)}.")

    # persistence: <workspace>/iterations/ is rewritten on every Save Workspace
    def _save_iterations(self):
        folder = os.path.join(self.workspace_dir, "iterations")
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        if not self.iterations:
            return
        os.makedirs(folder)
        meta = []
        for it in self.iterations:
            stem = os.path.join(folder, f"iter_{it['n']:03d}")
            if it["points"] is not None:
                it["points"].to_file(stem + "_points.gpkg", driver="GPKG")
            if it["lut"] is not None:
                np.save(stem + "_map.npy", it["lut"])
            if it["image"] is not None:
                it["image"].save(stem + "_matrix.png")
            meta.append({k: v for k, v in it.items() if k not in ("points", "lut", "image")})
        with open(os.path.join(folder, "iterations.json"), "w") as f:
            json.dump(meta, f)

    def _load_iterations(self):
        self.iterations = []
        folder = os.path.join(self.workspace_dir, "iterations")
        path = os.path.join(folder, "iterations.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    meta = json.load(f)
                for m in meta:
                    stem = os.path.join(folder, f"iter_{m['n']:03d}")
                    it = dict(m)
                    it["points"] = (self._read_accuracy_gpkg(stem + "_points.gpkg")
                                    if os.path.exists(stem + "_points.gpkg") else None)
                    it["lut"] = np.load(stem + "_map.npy") if os.path.exists(stem + "_map.npy") else None
                    img = QImage(stem + "_matrix.png") if os.path.exists(stem + "_matrix.png") else QImage()
                    it["image"] = None if img.isNull() else img
                    self.iterations.append(it)
                print(f" -> {len(self.iterations)} iteration(s) restored to the Iterations list.")
            except Exception as e:
                self.iterations = []
                print(f" -> Saved iterations could not be read ({e}).")
        self.refresh_iteration_table()

    @staticmethod
    def _read_accuracy_gpkg(path):
        acc_gdf = gpd.read_file(path)
        # GPKG round-trips a missing string field as NaN, not None -- normalize
        # back to None so truthy/identity checks elsewhere (which assume the
        # in-memory None convention used everywhere else this column is set)
        # don't misbehave on an unassigned point.
        if 'stratum' not in acc_gdf.columns:
            acc_gdf['stratum'] = None
        for col in ('reference_class', 'predicted_class', 'stratum'):
            acc_gdf[col] = acc_gdf[col].astype(object).where(acc_gdf[col].notna(), None)
        return acc_gdf

    def sync_classified_button_label(self):
        """ "Raw Ortho" until a classification exists, then "Classified Map" -- see
        the toggle_*_view methods below for what each state actually does. """
        has_classification = self.classified_image_data is not None
        self.cls_view_btn.setText("Classified Map" if has_classification else "Raw Ortho")

    def _sync_blend_availability(self):
        """ Blend Overlay is only ever selectable when a classification exists AND the
        Classified Map / Raw Ortho button is the active view -- toggling Segmented on
        always disables it, and it snaps back off (with the opacity controls) the
        moment it becomes unselectable, so it never gets left on in a state where it
        has nothing to blend. """
        has_classification = self.classified_image_data is not None
        blend_selectable = has_classification and self.cls_view_btn.isChecked() and not self.seg_view_btn.isChecked()
        self.blend_toggle_btn.setEnabled(blend_selectable)
        if not blend_selectable:
            self.blend_toggle_btn.setChecked(False)
            self.opacity_slider.setEnabled(False)
            self.opacity_spinbox.setEnabled(False)
            self.blend_mode_combo.setEnabled(False)

    def toggle_blend_mode(self):
        is_checked = self.blend_toggle_btn.isChecked()
        self.opacity_slider.setEnabled(is_checked)
        self.opacity_spinbox.setEnabled(is_checked)
        self.blend_mode_combo.setEnabled(is_checked)
        self.update_classification_display()

    def toggle_segmented_view(self):
        """ Segmented and Classified/Raw-Ortho are independently checkable (not a
        QButtonGroup radio pair) so Classified/Raw-Ortho can be toggled off by itself
        to fall back to the raw view -- see toggle_classified_view. Selecting Segmented
        always wins over the other button, and always kills Blend Overlay. """
        if self.seg_view_btn.isChecked():
            self.cls_view_btn.setChecked(False)
            self.current_view_mode = "segmented"
        else:
            self.current_view_mode = "classified" if (self.cls_view_btn.isChecked() and self.classified_image_data is not None) else "raw"
        self._sync_blend_availability()
        self.update_classification_display()

    def toggle_classified_view(self):
        """ Unlike Segmented, this button can be toggled OFF by itself (it's not
        exclusively grouped) -- doing so reveals the raw ortho (with its segment-
        boundary outline overlay), independent of whatever Segmented is doing. """
        if self.cls_view_btn.isChecked():
            self.seg_view_btn.setChecked(False)
            if self.classified_image_data is None:
                print("Notice: No classification result calculated yet. Displaying raw image instead.")
                self.current_view_mode = "raw"
            else:
                self.current_view_mode = "classified"
        else:
            self.current_view_mode = "raw"
        self._sync_blend_availability()
        self.update_classification_display()

    def setup_classification_view(self):
        """ After a fresh segmentation: Classification shows the segmented view (the
        app then moves on to the Discover tab, the next step). """
        if self.base_seg_qimg is None or self.raw_pixmap is None: return
        self.current_view_mode = "segmented"
        self.seg_view_btn.setChecked(True)
        self.cls_view_btn.setChecked(False)
        self.sync_classified_button_label()
        self._sync_blend_availability()
        self.class_viewer.set_image(self.raw_pixmap, grid_size=self.raw_grid)
        self._invalidate_display_caches()
        self.update_classification_display()

    def on_class_selection_changed(self):
        for i in range(self.class_list_widget.count()):
            item = self.class_list_widget.item(i)
            font = item.font()
            font.setBold(False)
            if not hasattr(item, '_original_size'):
                item._original_size = font.pointSize() if font.pointSize() > 0 else 10
            font.setPointSize(item._original_size)
            item.setFont(font)
            
        current_item = self.class_list_widget.currentItem()
        if current_item:
            font = current_item.font()
            font.setBold(True)
            if not hasattr(current_item, '_original_size'):
                current_item._original_size = font.pointSize() if font.pointSize() > 0 else 10
            font.setPointSize(current_item._original_size + 2)
            current_item.setFont(font)

    def toggle_samples_view(self):
        # ToggleSwitch paints its own on/off state -- no text to sync here.
        self.update_classification_display()

    def add_class(self):
        class_name = self.class_name_edit.text().strip()
        if not class_name or class_name in self.classes: return
        suggested = QColor(suggest_class_color(class_name, [d["color"].name() for d in self.classes.values()]))
        color = QColorDialog.getColor(suggested, self, f"Colour for '{class_name}' (suggested from its name)")
        if color.isValid():
            self.classes[class_name] = {"color": color, "training_segments": set()}
            item = QListWidgetItem(f"{class_name} (0)")
            item.setBackground(color)
            self.class_list_widget.addItem(item)
            self.class_name_edit.clear()
            self.rebuild_sample_class_map()
            self.refresh_accuracy_ui()
            print(f"Added structural class: {class_name}")

    def remove_class(self):
        current_item = self.class_list_widget.currentItem()
        if not current_item: return
        class_name = current_item.text().split(" (")[0]
        reply = QMessageBox.question(
            self, "Remove Class",
            f"Remove class '{class_name}' and all its training samples? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        self._delete_class(class_name)

    def _delete_class(self, class_name):
        """ Removes a class everywhere: training samples, the class list, Discover groups
        named after it (they become unnamed), and its areas on the classified map (they
        become unclassified; the remaining classes keep their colours). """
        old_order = list(self.classes)
        del self.classes[class_name]
        for i in range(self.class_list_widget.count()):
            if self.class_list_widget.item(i).text().split(" (")[0] == class_name:
                self.class_list_widget.takeItem(i)
                break
        self._reindex_classified(old_order, {class_name: None})
        if self.groups is not None:
            self.groups.rename_class(class_name, None)
            self.refresh_discover_list()
            self.update_discover_map()
        self.rebuild_sample_class_map()
        self.update_classification_display()
        self.refresh_accuracy_ui()
        print(f"Removed class: {class_name}")

    def _reindex_classified(self, old_order, name_map):
        """ The classified map stores each pixel's class as its position in self.classes.
        After classes are removed, renamed or merged, remap those positions (name_map: old
        name -> new name, or None for removed) and redraw the colours. """
        if self.raw_classified_raster_ints is None:
            return
        lut = np.zeros(max(len(old_order), 1) + 1, dtype=np.uint8)
        new_index = {n: i for i, n in enumerate(self.classes, start=1)}
        for i, n in enumerate(old_order, start=1):
            target = name_map.get(n, n)
            lut[i] = new_index.get(target, 0) if target else 0
        self.raw_classified_raster_ints = lut[self.raw_classified_raster_ints]
        if self.sieved_classified_raster_ints is not None:
            self.sieved_classified_raster_ints = lut[self.sieved_classified_raster_ints]
        self._recolor_classified()

    def _recolor_classified(self):
        """ Builds the classified-map display (display resolution, see display_stride) from
        the sieved class indices with the current class colours. classified_image_data is
        that small RGB array; export always uses the full-resolution indices. """
        if self.sieved_classified_raster_ints is None:
            return
        st = display_stride(self.sieved_classified_raster_ints.shape)
        disp = self.sieved_classified_raster_ints[::st, ::st]
        lut = np.zeros((max(len(self.classes), int(disp.max()) if disp.size else 0) + 1, 3), dtype=np.uint8)
        for idx, (name, data) in enumerate(self.classes.items(), start=1):
            c = data["color"]
            lut[idx] = (c.red(), c.green(), c.blue())
        self.classified_image_data = lut[disp]
        self.classified_qimg = bgrx_qimage(self.classified_image_data)

    def edit_class_dialog(self, item):
        old_name = item.text().split(" (")[0]
        old_color = self.classes[old_name]["color"]
        dialog = EditClassDialog(old_name, old_color, self)
        if dialog.exec() == QDialog.Accepted:
            new_name, new_color = dialog.get_values()
            if self._rename_class(old_name, new_name):
                self._set_class_color(new_name.strip(), new_color)

    def _rename_class(self, old_name, new_name):
        """ Renames a class everywhere (class list, accuracy points, Discover groups),
        keeping its position (the classified map stores positions). """
        new_name = (new_name or "").strip()
        if not new_name:
            print("Error: Class name cannot be empty.")
            return False
        if new_name == old_name:
            return True
        if new_name in self.classes:
            print(f"Error: A class named '{new_name}' already exists.")
            return False
        self.classes = {(new_name if k == old_name else k): v for k, v in self.classes.items()}
        for i in range(self.class_list_widget.count()):
            it = self.class_list_widget.item(i)
            if it.text().split(" (")[0] == old_name:
                it.setText(f"{new_name} ({len(self.classes[new_name]['training_segments'])})")
        # Accuracy points reference classes by name (not an ID like segments), so a
        # rename needs to follow through there too or those points silently orphan.
        if self.engine.accuracy_points_gdf is not None:
            gdf = self.engine.accuracy_points_gdf
            gdf.loc[gdf['reference_class'] == old_name, 'reference_class'] = new_name
            gdf.loc[gdf['predicted_class'] == old_name, 'predicted_class'] = new_name
        if self.groups is not None:
            self.groups.rename_class(old_name, new_name)
            self.refresh_discover_list()
            if self.disc_selected:
                self._refresh_name_combo()
                self._update_sel_header()
        print(f"Renamed class '{old_name}' to '{new_name}'.")
        self.refresh_accuracy_ui()
        return True

    def _set_class_color(self, name, color):
        """ Recolours a class everywhere and redraws what shows it, straight away. """
        color = QColor(color)
        if name not in self.classes or not color.isValid():
            return
        self.classes[name]["color"] = color
        for i in range(self.class_list_widget.count()):
            it = self.class_list_widget.item(i)
            if it.text().split(" (")[0] == name:
                it.setBackground(color)
        self._recolor_classified()
        self.rebuild_sample_class_map()
        self.update_classification_display()
        if self.groups is not None:
            self.refresh_discover_list()
            self.update_discover_map()
            if self.disc_selected:
                self._refresh_name_combo()
                self._update_sel_header()
        self.update_accuracy_map_display()
        self.refresh_accuracy_ui()

    def merge_selected_classes(self):
        selected_items = self.class_list_widget.selectedItems()
        if len(selected_items) < 2:
            print("Action Denied: You must select at least two classes to merge.")
            return
            
        class_names = [item.text().split(" (")[0] for item in selected_items]

        reply = QMessageBox.question(
            self, "Merge Classes",
            f"Merge {len(class_names)} classes ({', '.join(class_names)}) into one? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        current_item = self.class_list_widget.currentItem()
        default_name = current_item.text().split(" (")[0] if current_item in selected_items else class_names[-1]

        new_name, ok = QInputDialog.getText(
            self, 
            "Merge Classes", 
            f"Enter name for the merged class ({len(class_names)} selected):", 
            text=default_name
        )
        
        if not ok or not new_name.strip():
            return
            
        new_name = new_name.strip()
        
        old_order = list(self.classes)
        merged_segments = set()
        for name in class_names:
            merged_segments.update(self.classes[name]["training_segments"])
            
        merged_color = self.classes[default_name]["color"] if default_name in self.classes else QColor(Qt.gray)
        
        for name in class_names:
            if name in self.classes:
                del self.classes[name]
                
        self.classes[new_name] = {
            "color": merged_color,
            "training_segments": merged_segments
        }

        self.class_list_widget.clear()
        for name, data in self.classes.items():
            item = QListWidgetItem(f"{name} ({len(data['training_segments'])})")
            item.setBackground(data["color"])
            self.class_list_widget.addItem(item)

        # Same as a rename (see edit_class_dialog): accuracy points referencing any of
        # the merged-away class names by string need to follow the merge too.
        if self.engine.accuracy_points_gdf is not None:
            gdf = self.engine.accuracy_points_gdf
            merged_away = [n for n in class_names if n != new_name]
            if merged_away:
                gdf.loc[gdf['reference_class'].isin(merged_away), 'reference_class'] = new_name
                gdf.loc[gdf['predicted_class'].isin(merged_away), 'predicted_class'] = new_name

        self._reindex_classified(old_order, {n: new_name for n in class_names})
        if self.groups is not None:
            for n in class_names:
                self.groups.rename_class(n, new_name)
            self.refresh_discover_list()
            self.update_discover_map()
        self.rebuild_sample_class_map()
        print(f"Topological Merge Completed: Combined classes {class_names} into '{new_name}'.")
        self.update_classification_display()
        self.refresh_accuracy_ui()

    def collect_training_sample(self, pos):
        if self.engine.segmented_image is None: return
        segment_id = self.engine.segmented_image[pos.y(), pos.x()]
        if segment_id == 0: return
        
        existing_class = None
        for name, info in self.classes.items():
            if segment_id in info["training_segments"]:
                existing_class = name
                break
                
        if existing_class:
            self.classes[existing_class]["training_segments"].remove(segment_id)
            print(f"Deselection: Removed segment ID {segment_id} from '{existing_class}'.")
            
            self._update_sample_segment(int(segment_id))
            self.update_list_widget_counts()
            self.update_classification_display()
            return

        current_item = self.class_list_widget.currentItem()
        if not current_item:
            print("Attention: Please select a target class from the list panel before collecting samples.")
            return
            
        class_name = current_item.text().split(" (")[0]
        self.classes[class_name]["training_segments"].add(segment_id)
        
        self._update_sample_segment(int(segment_id))
        self.update_list_widget_counts()
        self.update_classification_display()

    def paint_training_sample(self, pos):
        """ Instantly registers selected points during a paint stroke brush drag. """
        if self.engine.segmented_image is None: return
        if not (0 <= pos.y() < self.engine.segmented_image.shape[0] and 0 <= pos.x() < self.engine.segmented_image.shape[1]):
            return
        segment_id = self.engine.segmented_image[pos.y(), pos.x()]
        if segment_id == 0: return
        
        current_item = self.class_list_widget.currentItem()
        if not current_item: return
        
        class_name = current_item.text().split(" (")[0]
        
        # Quick skip if this segment is already assigned to this exact class
        if segment_id in self.classes[class_name]["training_segments"]:
            return
            
        # Remove from other classes
        for name, data in self.classes.items():
            if segment_id in data["training_segments"]:
                data["training_segments"].remove(segment_id)
                
        self.classes[class_name]["training_segments"].add(segment_id)
        
        self._update_sample_segment(int(segment_id))
        self.update_classification_display()

    def paint_drag_finished(self):
        self.update_list_widget_counts()
        print("Completed continuous sample paint brush drag selection.")

    def toggle_selection_mode(self):
        self.selection_mode = self.selection_mode_btn.isChecked()
        self.selection_mode_btn.setText(f"Selection Mode: {'ON' if self.selection_mode else 'OFF'}")
        self.class_viewer.set_interaction_mode("select" if self.selection_mode else "paint")
        if not self.selection_mode:
            self.clear_segment_selection()
        print(f"Selection Mode {'enabled' if self.selection_mode else 'disabled'}.")

    def push_selection_undo(self):
        """ Snapshots locked_selection/current_seed_segments/tolerance before a mutating
        selection action, so Ctrl+Z can restore it. Capped to bound memory in long sessions. """
        self.selection_undo_stack.append((
            frozenset(self.locked_selection),
            frozenset(self.current_seed_segments),
            self.grow_tolerance_spinbox.value()
        ))
        if len(self.selection_undo_stack) > 50:
            self.selection_undo_stack.pop(0)

    def undo_selection(self):
        if not self.selection_undo_stack:
            print("Nothing to undo.")
            return
        locked, seeds, tolerance = self.selection_undo_stack.pop()
        self.locked_selection = set(locked)
        self.current_seed_segments = set(seeds)
        self._set_grow_tolerance_silently(tolerance)
        self.recompute_grown_selection()
        print("Undid last selection change.")

    def _set_grow_tolerance_silently(self, tolerance):
        """ Sets the tolerance slider/spinbox without triggering their cross-wired
        valueChanged callbacks (which would otherwise recompute the selection with
        stale locked/seed state before the caller has finished restoring it). """
        self.grow_tolerance_slider.blockSignals(True)
        self.grow_tolerance_spinbox.blockSignals(True)
        self.grow_tolerance_spinbox.setValue(tolerance)
        self.grow_tolerance_slider.setValue(int(tolerance * 100))
        self.grow_tolerance_slider.blockSignals(False)
        self.grow_tolerance_spinbox.blockSignals(False)

    def handle_selection_click(self, pos):
        if self.engine.segmented_image is None: return
        if not (0 <= pos.y() < self.engine.segmented_image.shape[0] and 0 <= pos.x() < self.engine.segmented_image.shape[1]):
            return
        seg_id = int(self.engine.segmented_image[pos.y(), pos.x()])
        if seg_id == 0: return

        self.push_selection_undo()

        if seg_id in self.selected_segments:
            self.locked_selection.discard(seg_id)
            self.current_seed_segments.discard(seg_id)
        else:
            # Lock in whatever is currently shown (including the previous click's grown
            # region), then start a fresh seed at zero tolerance -- so the slider only
            # ever grows the newest click outward, without re-expanding earlier picks.
            self.locked_selection |= self.selected_segments
            self.current_seed_segments = {seg_id}
            self._set_grow_tolerance_silently(0.0)

        self.recompute_grown_selection()

    def paint_selection_sample(self, pos):
        """ Ctrl+drag continuous add-to-selection, mirroring paint_training_sample's
        direct "add whatever's under the cursor" behavior -- no seed/tolerance growth,
        each painted-over segment goes straight into the locked selection. """
        if self.engine.segmented_image is None: return
        if not (0 <= pos.y() < self.engine.segmented_image.shape[0] and 0 <= pos.x() < self.engine.segmented_image.shape[1]):
            return
        seg_id = int(self.engine.segmented_image[pos.y(), pos.x()])
        if seg_id == 0: return
        if seg_id in self.selected_segments:
            return

        if not self._selection_drag_active:
            self.push_selection_undo()
            self._selection_drag_active = True

        self.locked_selection |= self.selected_segments
        self.current_seed_segments = set()
        self.locked_selection.add(seg_id)
        self.recompute_grown_selection()

    def selection_drag_finished(self):
        self._selection_drag_active = False
        print("Completed continuous selection paint brush drag.")

    def recompute_grown_selection(self):
        tolerance = self.grow_tolerance_spinbox.value()
        grown = self.engine.grow_selection(self.current_seed_segments, tolerance) if self.current_seed_segments else set()
        self.selected_segments = self.locked_selection | grown
        self.selection_count_label.setText(f"Selected: {len(self.selected_segments)} segments")
        self.update_classification_display()

    def on_grow_tolerance_changed(self, value):
        if self.selection_mode:
            self.recompute_grown_selection()

    def clear_segment_selection(self):
        self.push_selection_undo()
        self.selected_segments = set()
        self.locked_selection = set()
        self.current_seed_segments = set()
        self.selection_count_label.setText("Selected: 0 segments")
        self.update_classification_display()
        print("Selection cleared.")

    def assign_selection_to_class(self):
        if not self.selected_segments:
            print("Action Denied: No segments currently selected.")
            return
        current_item = self.class_list_widget.currentItem()
        if not current_item:
            print("Attention: Please select a target class from the list panel first.")
            return

        class_name = current_item.text().split(" (")[0]
        for seg_id in self.selected_segments:
            for name, data in self.classes.items():
                if name != class_name:
                    data["training_segments"].discard(seg_id)
            self.classes[class_name]["training_segments"].add(seg_id)

        print(f"Assigned {len(self.selected_segments)} selected segment(s) to class '{class_name}'.")
        self.rebuild_sample_class_map()
        self.update_list_widget_counts()
        self.clear_segment_selection()

    def merge_selection_into_object(self):
        if len(self.selected_segments) < 2:
            print("Merge Denied: Select at least two segments first (enable Selection Mode).")
            return

        result = self.engine.merge_segments(self.selected_segments)
        if result is None:
            print("Merge failed or was skipped; see warnings above.")
            return
        survivor_id, absorbed_ids = result
        print(f"Merged {len(absorbed_ids)} segment(s) {sorted(absorbed_ids)} into survivor segment {survivor_id}.")

        # Neighbor-relative features affect the survivor AND all its neighbors -- full
        # (cheap, dict-based) recompute rather than trying to enumerate the affected subset.
        self.engine.compute_neighbor_relative_features()

        # --- training_segments conflict resolution ---
        # If the survivor ID itself already belonged to a class, that class wins
        # unconditionally. Otherwise, the first non-empty class found among the merged
        # set wins (dict insertion order), and a warning is printed if the merged
        # segments spanned multiple different classes -- rather than silently guessing.
        absorbed_set = set(absorbed_ids)
        survivor_owner = None
        owning_classes = []
        for class_name, data in self.classes.items():
            if survivor_id in data["training_segments"]:
                survivor_owner = class_name
            if data["training_segments"] & absorbed_set:
                owning_classes.append(class_name)

        target_class = survivor_owner
        if target_class is None and owning_classes:
            target_class = owning_classes[0]
            if len(owning_classes) > 1:
                print(f"WARNING: Merged segments spanned multiple training classes {owning_classes}. "
                      f"Survivor segment {survivor_id} assigned to '{target_class}' (first match); "
                      f"training membership for classes {owning_classes[1:]} was dropped for the merged segments.")

        for class_name, data in self.classes.items():
            data["training_segments"] -= absorbed_set
            data["training_segments"].discard(survivor_id)
        if target_class is not None:
            self.classes[target_class]["training_segments"].add(survivor_id)

        self.selected_segments = set()
        self.locked_selection = set()
        self.current_seed_segments = set()
        self.selection_count_label.setText("Selected: 0 segments")
        # Absorbed IDs no longer exist -- any undo snapshot referencing them would be
        # restoring dead segment IDs, so history from before the merge is discarded.
        self.selection_undo_stack = []

        self.rebuild_sample_class_map()
        self.update_list_widget_counts()
        self.generate_boundary_and_visual_caches()  # survivor's mean color/shape changed
        self._discover_after_segment_change()
        self.engine.dsm_data = None   # survivor features are done; don't keep the DSM grid around
        self.update_classification_display()
        print("Adjacency graph and feature table refreshed post-merge.")

    def update_list_widget_counts(self):
        for i in range(self.class_list_widget.count()):
            item = self.class_list_widget.item(i)
            name = item.text().split(" (")[0]
            if name in self.classes:
                count = len(self.classes[name]["training_segments"])
                item.setText(f"{name} ({count})")

    def update_classification_display(self):
        """ The Classification map as layers over the shared ortho pixmap: the view layer
        (segment colours / classified map, at display resolution), segment edges in raw
        view, training samples (an array layer updated per segment, see
        _update_sample_segment) and the selection. Nothing full-size is copied, so a click
        stays fast on large images. """
        if self.engine.segmented_image is None or self.raw_pixmap is None: return
        v = self.class_viewer
        v.set_base_image_cached(self.raw_pixmap, grid_size=self.raw_grid)
        mode = self.current_view_mode
        if mode == "segmented":
            v.set_layer("view", self.base_seg_qimg, 1.0, z=1)
        elif mode == "classified" and self.classified_qimg is not None:
            opacity = 1.0
            if self.blend_toggle_btn.isChecked():
                opacity = self.opacity_slider.value() / 100.0 if self.opacity_toggle.isChecked() else 0.0
            multiply = self.blend_toggle_btn.isChecked() and self.blend_mode_combo.currentText() == "Multiply"
            v.set_layer("view", self.classified_qimg, opacity, z=1, multiply=multiply)
        else:
            v.set_layer("view", None)
        v.set_layer("edges", self.boundary_overlay_qimg if mode == "raw" else None, 1.0, z=2)
        if self._samples_item is None or v.layer("samples") is not self._samples_item:
            arr = self._samples_lut()[self._seg_disp()[1]]
            self._samples_item = v.set_array_layer("samples", arr, z=3)
        v.set_layer_opacity("samples", 1.0 if self.toggle_samples_btn.isChecked() else 0.0)
        self._update_selection_layer()

    def _seg_disp(self):
        """ (stride, segment raster at display resolution), cached until the segmentation changes. """
        if self._seg_disp_cache is None:
            seg = self.engine.segmented_image
            st = display_stride(seg.shape)
            self._seg_disp_cache = (st, np.ascontiguousarray(seg[::st, ::st]))
        return self._seg_disp_cache

    def _seg_slices(self):
        if self._seg_slices_cache is None:
            from scipy import ndimage
            self._seg_slices_cache = ndimage.find_objects(self.engine.segmented_image)
        return self._seg_slices_cache

    def _invalidate_display_caches(self):
        """ Call whenever segmented_image changes (new segmentation, workspace load, merge). """
        self._seg_disp_cache = None
        self._seg_slices_cache = None
        self._samples_item = None
        self._sel_key = None
        self._seg_px = None
        self.disc_focus_seg = None

    def _samples_lut(self):
        lut = np.zeros((int(self.engine.segmented_image.max()) + 1, 4), dtype=np.uint8)
        for name, data in self.classes.items():
            ids = np.fromiter((int(x) for x in data["training_segments"]), dtype=np.int64)
            ids = ids[(ids > 0) & (ids < len(lut))]
            lut[ids] = rgba_premultiplied(data["color"], 150)
        return lut

    def _update_sample_segment(self, segment_id):
        """ Repaints one segment in the samples layer (its class colour, or clear). """
        item = self._samples_item
        sl = self._seg_slices()
        if item is None or segment_id <= 0 or segment_id > len(sl) or sl[segment_id - 1] is None:
            return
        cls = self._class_of_segment(int(segment_id))
        rgba = rgba_premultiplied(self.classes[cls]["color"], 150) if cls else (0, 0, 0, 0)
        st, seg_d = self._seg_disp()
        rs, cs = sl[segment_id - 1]
        r0, r1 = rs.start // st, -(-rs.stop // st)
        c0, c1 = cs.start // st, -(-cs.stop // st)
        sub = seg_d[r0:r1, c0:c1] == segment_id
        item.arr[r0:r1, c0:c1][sub] = rgba
        item.update_region(r0, r1, c0, c1)

    def _update_selection_layer(self):
        key = frozenset(self.selected_segments)
        if key == self._sel_key and (not key or self.class_viewer.layer("selection") is not None):
            return
        self._sel_key = key
        if not key:
            self.class_viewer.set_layer("selection", None)
            return
        lut = np.zeros((int(self.engine.segmented_image.max()) + 1, 4), dtype=np.uint8)
        lut[np.fromiter(key, dtype=np.int64)] = rgba_premultiplied(QColor(0, 255, 255), 110)
        self.class_viewer.set_array_layer("selection", lut[self._seg_disp()[1]], z=4)

    def reload_dsm_singly(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select DSM Model to Reload Singly", "", "GeoTIFF (*.tif *.tiff)")
        if file_path:
            self.dsm_path_edit.setText(file_path)
            try:
                print("Reloading DSM model on-the-fly from step 3...")
                self.engine.load_dsm(file_path)
                print("DSM re-integrated and features recalculated successfully.")
                self.update_dsm_status_label()
                self.rebuild_hillshade_cache()

                print("Recalculating segment features to incorporate DSM variables...")
                self.engine.compute_all_features_parallel()
                self.engine.compute_neighbor_relative_features()
                print(" -> Segmentation feature table updated with DSM metrics.")
            except Exception as e:
                print(f"Failed to reload DSM: {e}")

    # --- Classifier features (19 base, + DSM, + deep appearance) ---
    def _prepare_classifier_features(self):
        """ Sets whether the classifier uses the deep_* columns, adding them from the cached
        deep features when needed. Returns False if they were asked for but aren't available. """
        want = self.use_deep_check.isChecked()
        self.engine.use_deep_features = False
        if not want:
            return True
        deep = self._deep_for_current()
        if deep is None:
            print("Deep appearance features are not computed for this segmentation yet: in Discover Classes, "
                  "tick Appearance and press Find Groups once, then try again (or untick "
                  "'Use deep appearance features').")
            return False
        have = [c for c in self.engine.features_df.columns if c.startswith(self.engine.DEEP_PREFIX)]
        if not have or self.engine.features_df[have].abs().sum().sum() == 0:
            print("Adding deep appearance features (PCA of DINOv2 5 m and 10 m patches) to the classifier features...")
            QApplication.processEvents()
            self.engine.add_deep_columns(deep)
        self.engine.use_deep_features = True
        return True

    def train_classifier(self):
        print("Extracting sample structural features for training...")
        QApplication.processEvents()
        if not self._prepare_classifier_features():
            return
        
        training_samples = []
        for class_name, data in self.classes.items():
            for seg_id in data["training_segments"]:
                training_samples.append({"class": class_name, "raster_val": seg_id})
        
        if len(training_samples) < 2 or len(self.classes) < 2:
            print("Training Error: Select samples for at least two classes first.")
            return
            
        training_df = pd.DataFrame(training_samples)
        final_training_df = pd.merge(self.engine.features_df, training_df, on='raster_val').fillna(0)
        
        features = self.engine.active_feature_columns()
        self.classifier_features = list(features)
        n_deep = sum(c.startswith(self.engine.DEEP_PREFIX) for c in features)
        print(f" -> {len(features)} features: {len(features) - n_deep} spectral/texture/shape"
              f"{'/elevation' if self.engine.has_dsm else ''}"
              f"{f' + {n_deep} deep appearance' if n_deep else ''}: {features}")
        
        X = final_training_df[features].values
        y = final_training_df['class'].values
        
        print(f"Fitting Random Forest with 100 decision trees over {len(X)} active samples...")
        QApplication.processEvents()
        
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X, y)
        print("Random Forest model trained successfully.")

    def check_class_separability(self):
        """ Jeffries-Matusita separability of the current training samples, shown as a
        figure (also saved as class_separability.png in the workspace). Meant for
        before training: pairs rated fair/poor are the ones to add samples to, check,
        or merge. """
        if self.engine.features_df is None:
            print("Separability: segment the image and compute features first.")
            return
        if not self._prepare_classifier_features():
            return
        training = {name: list(data["training_segments"]) for name, data in self.classes.items()}
        result = self.engine.compute_class_separability(training)
        if len(result["classes"]) < 2:
            print("Separability: at least two classes need 2 or more training samples each.")
            return
        print("Class separability (Jeffries-Matusita, 0-2):")
        for (a, b) in sorted(result["jm_feature"], key=lambda k: result["jm"].loc[k[0], k[1]]):
            print(f" -> {a} vs {b}: {result['jm'].loc[a, b]:.3f}")
        class_colors = {name: data["color"] for name, data in self.classes.items()}
        png_path = None
        if self.workspace_dir:
            self._ensure_workspace_dir()
            png_path = os.path.join(self.workspace_dir, "class_separability.png")
        self.show_separability_dialog(result, class_colors, png_path)

    def show_separability_dialog(self, result, class_colors, png_path):
        """ Separability figure with a pair selector: the signature plot's SD bands follow
        the chosen pair (default: the closest one). """
        dialog = QDialog(self)
        dialog.setWindowTitle("Class Separability")
        layout = QVBoxLayout(dialog)

        pairs = sorted(result["jm_feature"], key=lambda k: result["jm"].loc[k[0], k[1]])
        pair_row = QHBoxLayout()
        pair_row.addWidget(QLabel("Signature plot pair:"))
        pair_combo = QComboBox()
        for a, b in pairs:
            pair_combo.addItem(f"{a} vs {b}  ({result['jm'].loc[a, b]:.2f})", (a, b))
        pair_row.addWidget(pair_combo)
        pair_row.addStretch(1)
        layout.addLayout(pair_row)

        label = QLabel()
        scroll = QScrollArea()
        scroll.setWidget(label)
        scroll.setAlignment(Qt.AlignCenter)
        layout.addWidget(scroll)
        path_label = QLabel()
        path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        path_label.setStyleSheet("color: #76817b;")
        layout.addWidget(path_label)
        state = {}

        def render():
            image = render_separability_image(result, class_colors, focus_pair=pair_combo.currentData())
            state["image"] = image
            pixmap = QPixmap.fromImage(image)
            pixmap.setDevicePixelRatio(2)
            label.setPixmap(pixmap)
            label.resize(image.width() // 2, image.height() // 2)
            if png_path and image.save(png_path):
                path_label.setText(f"Saved to: {png_path}")
            elif not png_path:
                path_label.setText("Not saved (no workspace yet) -- use Copy Image.")

        render()
        pair_combo.currentIndexChanged.connect(lambda _: render())
        image = state["image"]
        screen = self.screen().availableGeometry()
        scroll.setMinimumSize(min(image.width() // 2 + 4, screen.width() - 120),
                              min(image.height() // 2 + 4, screen.height() - 240))

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        copy_btn = QPushButton("Copy Image")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setImage(state["image"]))
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        buttons.addWidget(copy_btn)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)
        dialog.exec()

    # --- Classification tab: find similar, what to label next ---
    def _similarity_space(self):
        """ (ids, X, id -> row) to measure "looks like" in: the Discover groups' space if
        groups exist, else deep + classifier features if deep features are cached, else
        the classifier features alone. """
        if self.groups is not None:
            return self.groups.ids, self.groups.X, self.groups.pos
        if self._sim_space is None:
            ids = np.array([s for s in self._segment_index()[0] if s in self.engine.features_df.index])
            deep = self._deep_for_current()
            X = od.build_space(ids, self.engine.features_df, self.engine.active_feature_columns(include_deep=False),
                               deep, deep is not None, True)
            self._sim_space = (ids, X, {int(s): i for i, s in enumerate(ids)})
        return self._sim_space

    def _active_class_name(self):
        item = self.class_list_widget.currentItem()
        return item.text().split(" (")[0] if item else (next(iter(self.classes)) if self.classes else None)

    def _class_of_segment(self, sid):
        for name, data in self.classes.items():
            if sid in data["training_segments"]:
                return name
        return None

    def on_class_viewer_context_menu(self, pixmap_pos, global_pos):
        sid = self._segment_at(pixmap_pos)
        if sid <= 0 or self.engine.features_df is None:
            return
        menu = QMenu(self)
        cls = self._class_of_segment(sid)
        head = menu.addAction(f"Segment #{sid}" + (f" · {cls} (sample)" if cls else ""))
        head.setEnabled(False)
        menu.addSeparator()
        act_similar = menu.addAction("Find Similar Segments")
        act_group = None
        if self.groups is not None and self.groups.pos.get(sid) is not None and self.groups.labels[self.groups.pos[sid]]:
            act_group = menu.addAction(f"Show Its Group ({self.groups.labels[self.groups.pos[sid]]}, Discover tab)")
        chosen = menu.exec(global_pos)
        if chosen is act_similar:
            self.find_similar_from_map(sid)
        elif act_group is not None and chosen is act_group:
            self.tabs.setCurrentWidget(self.discover_tab)
            self.disc_filter_combo.setCurrentIndex(0)
            self.disc_multi = []
            self.show_discover_group(self.groups.labels[self.groups.pos[sid]])

    def find_similar_from_map(self, sid):
        if not self.classes:
            print("Find Similar: add a class first.")
            return
        ids, X, pos = self._similarity_space()
        if sid not in pos:
            return
        idx = np.argsort(np.linalg.norm(X - X[pos[sid]], axis=1))[1:21]
        chips = [self._chip(ids[i], 120) for i in idx]
        caps = []
        for i in idx:
            c = self._class_of_segment(int(ids[i]))
            caps.append(f"#{int(ids[i])}" + (f" · {c}" if c else ""))
        default = self._class_of_segment(sid) or self._active_class_name()
        dlg = ChipPickerDialog(
            "Similar Segments", f"The 20 segments that look most like #{sid}, most similar first. "
            f"Tick the ones of one class and Add; repeat for other classes; Close when done.",
            chips, caps, list(self.classes), "Add ticked to", "Add", default_target=default, parent=self,
            on_apply=lambda picked, cls: self._add_samples({int(ids[idx[i]]) for i in picked}, cls, "Find Similar"))
        dlg.exec()

    def show_label_next(self):
        """ Segments the classifier is least sure about, spread over different kinds of
        segment (one or two per Discover group, or per appearance cluster), so labelling
        them teaches the classifier the most. """
        if self.classifier is None:
            print("What to Label Next: train the classifier first.")
            return
        fdf = self.engine.features_df
        cols = getattr(self, "classifier_features", None) or self.engine.active_feature_columns()
        proba = self.classifier.predict_proba(fdf[cols].fillna(0).values)
        names = list(self.classifier.classes_)
        top2 = np.sort(proba, axis=1)[:, -2:] if proba.shape[1] > 1 else np.c_[np.zeros(len(proba)), proba[:, -1]]
        margin = top2[:, 1] - top2[:, 0]
        seg_ids = fdf.index.values.astype(int)
        sampled = set().union(*[d["training_segments"] for d in self.classes.values()]) if self.classes else set()
        order = [i for i in np.argsort(margin) if int(seg_ids[i]) not in sampled][:600]
        ids, X, pos = self._similarity_space()
        picked = []
        if self.groups is not None:
            per_group = {}
            for i in order:
                j = self.groups.pos.get(int(seg_ids[i]))
                key = self.groups.labels[j] if j is not None else ""
                if not key or self.groups.groups[key]["status"] == "ignored":
                    continue
                if per_group.get(key, 0) < 2:
                    per_group[key] = per_group.get(key, 0) + 1
                    picked.append(i)
                if len(picked) >= 20:
                    break
        else:
            from sklearn.cluster import KMeans
            cand = [i for i in order[:300] if int(seg_ids[i]) in pos]
            if len(cand) > 20:
                Z = X[[pos[int(seg_ids[i])] for i in cand]]
                lab = KMeans(20, n_init=2, random_state=0).fit_predict(Z)
                for c in range(20):
                    members = [cand[k] for k in np.where(lab == c)[0]]
                    picked.append(min(members, key=lambda i: margin[i]))
            else:
                picked = cand
        if not picked:
            print("What to Label Next: nothing to suggest.")
            return
        chips = [self._chip(int(seg_ids[i]), 120) for i in picked]
        caps = []
        for i in picked:
            best = np.argsort(proba[i])[::-1][:2]
            caps.append(" · ".join(f"{names[b]} {100 * proba[i, b]:.0f}%" for b in best))
        dlg = ChipPickerDialog(
            "What to Label Next", "Segments the classifier is least sure about, spread over different kinds of "
            "segment. Tick the ones of one class and Add -- they leave the list; repeat for other classes, "
            "then Close and retrain.",
            chips, caps, list(self.classes), "Add ticked to", "Add", default_target=self._active_class_name(),
            ticked=False, parent=self,
            on_apply=lambda ks, cls: self._add_samples({int(seg_ids[picked[k]]) for k in ks}, cls, "What to Label Next"))
        dlg.exec()

    def _add_samples(self, sids, cls, source):
        if not sids or cls not in self.classes:
            return
        for other in self.classes.values():
            other["training_segments"] -= sids
        self.classes[cls]["training_segments"] |= sids
        self.rebuild_sample_class_map()
        self.update_list_widget_counts()
        self.update_classification_display()
        print(f"{source}: added {len(sids)} segment(s) to '{cls}'.")

    def classify_all(self):
        if self.classifier is None:
            print("Classification Error: Train the classifier model first.")
            return
            
        print("Evaluating predictions via trained Random Forest estimators...")
        QApplication.processEvents()
        
        features = getattr(self, "classifier_features", None) or self.engine.active_feature_columns()
        missing = [c for c in features if c not in self.engine.features_df.columns]
        if missing:
            print(f"Classification Error: the classifier was trained with features this workspace no longer has "
                  f"({', '.join(missing[:3])}...). Train again.")
            return
        predictions = self.classifier.predict(self.engine.features_df[features].fillna(0).values)
        seg_id_to_class = dict(zip(self.engine.features_df.index, predictions))
        
        h, w = self.engine.segmented_image.shape
        self.raw_classified_raster_ints = np.zeros((h, w), dtype=np.uint8)
        
        class_mapping = {class_name: idx for idx, class_name in enumerate(self.classes.keys(), start=1)}
        
        for class_name, data in self.classes.items():
            class_idx = class_mapping[class_name]
            predicted_seg_ids = [seg_id for seg_id, pred_class in seg_id_to_class.items() if pred_class == class_name]
            if not predicted_seg_ids: continue
            
            mask = np.isin(self.engine.segmented_image, predicted_seg_ids)
            self.raw_classified_raster_ints[mask] = class_idx

        self.sieved_classified_raster_ints = self.raw_classified_raster_ints.copy()
        self.apply_generalization_sieve()

    def apply_generalization_sieve(self):
        if self.raw_classified_raster_ints is None: return
        threshold = self.gen_spinbox.value()
        
        print(f"Applying Spatial Generalization Sieve (MMU: {threshold} px)...")
        QApplication.processEvents()
        
        self.sieved_classified_raster_ints = self.engine.apply_sieve_filter(self.raw_classified_raster_ints, threshold)
        
        self._recolor_classified()

        self.current_view_mode = "classified"
        self.sync_classified_button_label()
        self.seg_view_btn.setChecked(False)
        self.cls_view_btn.setChecked(True)
        self._sync_blend_availability()
        self.update_classification_display()
        print("Sieving completed successfully.")

    def export_classified_tif(self):
        if self.classified_image_data is None:
            print("Export Error: Run classification first.")
            return
            
        base_name = "map"
        if self.engine.ortho_path:
            _, file_name = os.path.split(self.engine.ortho_path)
            base_name, _ = os.path.splitext(file_name)
            
        default_path = os.path.join(self.workspace_dir, f"{base_name}_classified.tif")
        export_path, _ = QFileDialog.getSaveFileName(self, "Export Spatial Classified Map", default_path, "GeoTIFF (*.tif)")
        if not export_path: return
        
        try:
            print(f"Writing single-band classified GeoTIFF directly to: {export_path}")
            QApplication.processEvents()
            
            profile = self.engine.ortho_meta.copy()
            profile.update(
                dtype=rasterio.uint8,
                count=1,
                nodata=0
            )
            
            with rasterio.open(export_path, 'w', **profile) as dst:
                dst.write(self.sieved_classified_raster_ints, 1)
                
                colormap = {0: (0, 0, 0, 0)}
                for idx, (class_name, data) in enumerate(self.classes.items(), start=1):
                    color = data["color"]
                    colormap[idx] = (color.red(), color.green(), color.blue(), 255)
                dst.write_colormap(1, colormap)
                
            qml_path = os.path.splitext(export_path)[0] + ".qml"
            self.engine.write_qgis_styling(qml_path, self.classes)
            print("Export completed cleanly. Sidecar QML generated.")
        except Exception as e:
            print(f"Export failed: {e}")

    def save_workspace_session(self):
        if self.workspace_dir is None:
            print("Error: Load an orthomosaic first.")
            return
            
        try:
            print("Saving Workspace components to disk cache folder...")
            QApplication.processEvents()
            self._ensure_workspace_dir()

            seg_tif_path = os.path.join(self.workspace_dir, "segments.tif")
            profile = self.engine.ortho_meta.copy()
            profile.update(dtype=rasterio.int32, count=1, nodata=0)
            with rasterio.open(seg_tif_path, 'w', **profile) as dst:
                dst.write(self.engine.segmented_image, 1)
                
            self.engine.segments_gdf.to_file(os.path.join(self.workspace_dir, "segments.gpkg"), driver="GPKG")
            self.engine.features_df.to_csv(os.path.join(self.workspace_dir, "segment_features.csv"))
            
            if self.raw_classified_raster_ints is not None:
                np.save(os.path.join(self.workspace_dir, "classified_map.npy"), self.raw_classified_raster_ints)

            # Accuracy points survive a close without requiring "Run Accuracy
            # Assessment" to have been pressed -- a long labeling session across
            # hundreds of points is real work, and Execute shouldn't be the only save
            # point. This writes to the same accuracy_points.gpkg path Run Accuracy
            # Assessment itself writes at the end of a run, so there's one file, not
            # two competing copies.
            if self.engine.accuracy_points_gdf is not None and len(self.engine.accuracy_points_gdf) > 0:
                self.engine.accuracy_points_gdf.to_file(os.path.join(self.workspace_dir, "accuracy_points.gpkg"), driver="GPKG")

            session_data = {
                "n_segments": self.n_segments_spinbox.value(),
                "compactness": self.compactness_dspinbox.value(),
                "sigma": self.sigma_dspinbox.value(),
                "ortho_path": self.engine.ortho_path,
                "effective_pixel_size": self.engine.effective_pixel_size,
                "feature_pixel_size": self.engine.feature_pixel_size,
                "dsm_path": self.engine.dsm_path,
                "accuracy_sample_size": self.acc_sample_size_spinbox.value(),
                "accuracy_next_point_id": self.engine.next_accuracy_point_id,
                "accuracy_design": self.acc_design_combo.currentIndex(),
                "accuracy_min_per_class": self.acc_min_per_class_spinbox.value(),
                "accuracy_strata": self.engine.accuracy_strata,
                "use_deep_in_classifier": self.use_deep_check.isChecked(),
                "classes": {}
            }
            for class_name, data in self.classes.items():
                session_data["classes"][class_name] = {
                    "color": data["color"].name(),
                    "training_segments": [int(x) for x in data["training_segments"]]
                }
                
            with open(os.path.join(self.workspace_dir, "session.json"), 'w') as f:
                json.dump(session_data, f, indent=4)

            self._discover_save_to_workspace()
            self._save_iterations()
            self._mark_saved()
                
            print("Workspace saved successfully.")
        except Exception as e:
            print(f"Save failed: {e}")

    def _locate_missing_raster(self, description, missing_path):
        """ A workspace's session.json records absolute source paths (ortho, DSM) --
        if the referenced file has moved/been renamed/lives on a different machine
        since the workspace was saved, offer to browse for its new location instead
        of just failing the load. Returns the resolved path, or None if the user
        declines or cancels. """
        shown_path = missing_path or "(no path recorded)"
        reply = QMessageBox.question(
            self, "Locate Missing File",
            f"This workspace's recorded {description} could not be found:\n{shown_path}\n\n"
            f"Would you like to locate it manually?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        if reply != QMessageBox.Yes:
            return None
        start_dir = ""
        if missing_path:
            candidate_dir = os.path.dirname(missing_path)
            if os.path.isdir(candidate_dir):
                start_dir = candidate_dir
        new_path, _ = QFileDialog.getOpenFileName(
            self, f"Locate {description}", start_dir,
            "GeoTIFF Files (*.tif *.tiff);;All Files (*)"
        )
        return new_path or None

    def load_workspace_session(self):
        target_workspace_dir = self.workspace_dir
        if target_workspace_dir is None:
            # No ortho loaded yet this session -- browse straight to a saved workspace
            # folder instead of requiring the ortho to already be loaded (and its
            # resample manually re-applied to match) first.
            target_workspace_dir = QFileDialog.getExistingDirectory(self, "Select Workspace Folder")
            if not target_workspace_dir:
                return

        session_json = os.path.join(target_workspace_dir, "session.json")
        if not os.path.exists(session_json):
            print("Error: No saved session.json found in workspace.")
            return

        try:
            print("Reconstructing session workspace elements...")
            QApplication.processEvents()

            with open(session_json, 'r') as f:
                session_data = json.load(f)

            # Fetch the ortho this workspace was built from if it isn't already loaded,
            # then resample it to whatever resolution the workspace was saved at --
            # segments.tif/features (read further below) were computed against that
            # exact pixel grid and won't line up with a differently-sized ortho.
            with rasterio.open(os.path.join(target_workspace_dir, "segments.tif")) as src:
                saved_shape = (src.height, src.width)
            if self.engine.ortho_data is None:
                ortho_path = session_data.get("ortho_path")
                if not ortho_path or not os.path.exists(ortho_path):
                    ortho_path = self._locate_missing_raster("orthomosaic", ortho_path)
                    if not ortho_path:
                        print("Error: Orthomosaic not found and none was located -- workspace load cancelled.")
                        return
                self.path_edit.setText(ortho_path)
                self.load_ortho(out_shape=saved_shape)   # straight onto the saved grid
                if self.engine.ortho_data is None:
                    print("Error: Failed to load the workspace's orthomosaic.")
                    return
                # load_ortho() re-derives workspace_dir from the ortho path using the
                # same naming convention save_workspace_session relies on -- pin it back
                # to the folder session.json actually came from either way, since every
                # read below goes through self.workspace_dir.
                self.workspace_dir = target_workspace_dir

            target_res = session_data.get("effective_pixel_size")
            # Match the saved segments.tif grid exactly: recomputing the size from the
            # pixel size can round one row/column differently from the run that saved it.
            if self.engine.ortho_data.shape[1:] != saved_shape:
                print(f"Resampling orthomosaic to match this workspace's saved grid "
                      f"({saved_shape[1]}x{saved_shape[0]} px, ~{target_res:.4f} units/px)...")
                QApplication.processEvents()
                self.engine.resample_ortho(out_shape=saved_shape)
                target_res = self.engine.effective_pixel_size
                self.reset_colors()
                self.viewer.fit_view()
                self.acc_viewer.fit_view()

                # Reflect the applied resample back onto the resample controls so a
                # later "Apply Resample" click starts from the resolution actually in
                # memory, not the 100%/native default load_ortho() just reset them to.
                self._syncing_resample_controls = True
                if self.engine.native_crs_is_geographic:
                    scale = max(5, min(100, round(100.0 * self.engine.native_pixel_size / target_res)))
                    self.resample_scale_slider.setValue(scale)
                    self.resample_scale_spinbox.setValue(scale)
                else:
                    self.resample_target_res_spinbox.setValue(target_res * 100.0)
                    scale = max(5, min(100, round(100.0 * self.engine.native_pixel_size / target_res)))
                    self.resample_scale_slider.setValue(scale)
                    self.resample_scale_spinbox.setValue(scale)
                self._syncing_resample_controls = False
                self.update_resample_preview_label()

            self.n_segments_spinbox.setValue(session_data["n_segments"])
            self.compactness_dspinbox.setValue(session_data["compactness"])
            self.sigma_dspinbox.setValue(session_data["sigma"])
            self._configure_feature_res_control()
            fp = session_data.get("feature_pixel_size")
            self.engine.feature_pixel_size = fp
            if fp:
                self.feature_res_spinbox.setValue(fp * 100.0)
            
            if session_data.get("dsm_path"):
                dsm_p = session_data["dsm_path"]
                if not os.path.exists(dsm_p):
                    dsm_p = self._locate_missing_raster("DSM/elevation model", dsm_p)
                if dsm_p and os.path.exists(dsm_p):
                    self.dsm_path_edit.setText(dsm_p)
                    self.engine.load_dsm(dsm_p)
                    self.update_dsm_status_label()
                    self.rebuild_hillshade_cache()
                else:
                    print("Notice: DSM not found and none was located -- continuing without elevation data.")

            with rasterio.open(os.path.join(self.workspace_dir, "segments.tif")) as src:
                self.engine.segmented_image = src.read(1).astype(np.int32)

            self.engine.build_adjacency_graph()  # transient, not persisted -- rebuild on every load

            # A loaded segmented_image is a different raster than whatever was selected
            # in this session, if any -- old segment IDs (and undo history) don't apply.
            self.selected_segments = set()
            self.locked_selection = set()
            self.current_seed_segments = set()
            self.selection_undo_stack = []
            self.selection_count_label.setText("Selected: 0 segments")

            self.engine.segments_gdf = gpd.read_file(os.path.join(self.workspace_dir, "segments.gpkg"))
            self.engine.features_df = pd.read_csv(os.path.join(self.workspace_dir, "segment_features.csv")).set_index('raster_val')
            
            needs_slope_upgrade = self.engine.has_dsm and 'slope_mean' not in self.engine.features_df.columns
            needs_neighbor_upgrade = (
                'brightness_rel_neighbors' not in self.engine.features_df.columns or
                (self.engine.has_dsm and 'dsm_rel_neighbors' not in self.engine.features_df.columns)
            )
            if self.engine.has_dsm and 'hag_mean' not in self.engine.features_df.columns:
                print("Adding height above ground to this workspace's features...")
                QApplication.processEvents()
                self.engine.compute_height_features()
                self.engine.features_df.to_csv(os.path.join(self.workspace_dir, "segment_features.csv"))
            if 'contrast' not in self.engine.features_df.columns or needs_slope_upgrade or needs_neighbor_upgrade:
                print("Older workspace schema detected. Upgrading feature layers...")
                QApplication.processEvents()
                self.engine.compute_all_features_parallel()
                self.engine.compute_neighbor_relative_features()
                self.engine.features_df.to_csv(os.path.join(self.workspace_dir, "segment_features.csv"))
                
            self.classes = {}
            self.class_list_widget.clear()
            for class_name, data in session_data["classes"].items():
                color = QColor(data["color"])
                self.classes[class_name] = {
                    "color": color,
                    "training_segments": set(data["training_segments"])
                }
                item = QListWidgetItem(f"{class_name} ({len(data['training_segments'])})")
                item.setBackground(color)
                self.class_list_widget.addItem(item)
                
            self.rebuild_sample_class_map()  # Populate the fast index lookup map
            self.generate_boundary_and_visual_caches()
            self.show_boundaries_check.setEnabled(True)
            self.refresh_load_view_overlay()
            self._discover_load_from_workspace()

            map_npy = os.path.join(self.workspace_dir, "classified_map.npy")
            if os.path.exists(map_npy):
                self.raw_classified_raster_ints = np.load(map_npy)
                self.sieved_classified_raster_ints = self.raw_classified_raster_ints.copy()
                self.apply_generalization_sieve()
            else:
                self.setup_classification_view()

            # Accuracy Assessment (Tab 3) state
            self.acc_sample_size_spinbox.setValue(session_data.get("accuracy_sample_size", 200))
            self.engine.next_accuracy_point_id = session_data.get("accuracy_next_point_id", 1)
            self.acc_design_combo.setCurrentIndex(session_data.get("accuracy_design", 0))
            self.acc_min_per_class_spinbox.setValue(session_data.get("accuracy_min_per_class", 50))
            self.engine.accuracy_strata = session_data.get("accuracy_strata")
            self.use_deep_check.setChecked(bool(session_data.get("use_deep_in_classifier", False)))
            acc_gpkg = os.path.join(self.workspace_dir, "accuracy_points.gpkg")
            if os.path.exists(acc_gpkg):
                self.engine.accuracy_points_gdf = self._read_accuracy_gpkg(acc_gpkg)
            else:
                self.engine.accuracy_points_gdf = None
            self.acc_checked_point_ids = set()
            self.acc_nav_index = -1
            self.refresh_accuracy_ui()
            self._load_iterations()

            # Fill the map window at maximum zoom rather than leaving whatever
            # zoom/pan a previous image (or the initial blank state) left behind.
            self.viewer.fit_view()
            self.acc_viewer.fit_view()

            print("Workspace session successfully restored.")
            self._mark_saved()
        except Exception as e:
            print(f"Restore failed: {e}")

    def clear_training_data(self):
        if not self.classes:
            return
        reply = QMessageBox.question(
            self, "Clear All Training Data",
            "Clear all training samples for every class? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        for class_name in self.classes:
            self.classes[class_name]["training_segments"] = set()
        self.rebuild_sample_class_map()
        self.update_list_widget_counts()
        self.update_classification_display()
        print("Cleared sample collections.")

    def clear_selected_class_training_data(self):
        selected_items = self.class_list_widget.selectedItems()
        if not selected_items:
            print("Action Denied: Select at least one class from the list to clear its samples.")
            return

        class_names = [item.text().split(" (")[0] for item in selected_items]

        reply = QMessageBox.question(
            self, "Clear Selected Class",
            f"Clear all training samples for: {', '.join(class_names)}? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        for class_name in class_names:
            if class_name in self.classes:
                self.classes[class_name]["training_segments"] = set()

        self.rebuild_sample_class_map()
        self.update_list_widget_counts()
        self.update_classification_display()
        print(f"Cleared sample collections for: {class_names}")

    def unload_surface_data(self):
        if not self.engine.has_dsm:
            print("Notice: No surface data (DSM) is currently loaded.")
            return
        
        self.engine.dsm_data = None
        self.engine.slope_data = None
        self.engine.dsm_path = None
        self.dsm_path_edit.clear()

        if self.engine.features_df is not None:
            cols_to_drop = ['dsm_mean', 'dsm_std', 'slope_mean', 'dsm_rel_neighbors']
            existing_cols = [c for c in cols_to_drop if c in self.engine.features_df.columns]
            if existing_cols:
                self.engine.features_df.drop(columns=existing_cols, inplace=True)
                print(f"Removed surface structural components from active feature matrix: {existing_cols}")
                
        self.update_dsm_status_label()
        self.rebuild_hillshade_cache()
        print("Surface data (DSM) successfully unloaded. Classification features reverted to 2D spectral/texture metrics.")

    def update_dsm_status_label(self):
        status_text = "DSM Status: ACTIVE (Elevation maps included)" if self.engine.has_dsm else "DSM Status: INACTIVE (Spectral maps only)"
        self.dsm_status_label.setText(status_text)
        print(f" -> System feature footprint updated: {status_text}")

    def reset_interface(self):
        """ Full session reset: clears the loaded image/DSM, segmentation, training
        samples, and any classification result, then returns to the Load/View/Segment
        tab. Does NOT touch anything already written to disk by Save Workspace Session
        -- only in-memory/on-screen state. Loading a new orthomosaic afterward computes
        a fresh workspace_dir (see load_ortho), so a subsequent Save Workspace Session
        writes to that new image's own folder rather than overwriting the previous
        session's saved outputs. """
        reply = QMessageBox.question(
            self, "Reset and Return to Loading",
            "Reset the entire session? This clears the loaded image, DSM, segmentation, "
            "training samples, and any classification result. Anything not already saved "
            "via Save Workspace Session will be lost.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        self._clear_session()

    def _clear_session(self):
        """ Clears everything in memory and returns to Tab 1 (no prompt). """
        self.engine = ObiaEngine()
        self.workspace_dir = None
        self.classes = {}
        self.classifier = None

        self.base_raw_qimg = None
        self.raw_pixmap = None
        self.raw_grid = None          # (w, h) of the working grid the base pixmap stands for
        self.classified_qimg = None
        self._seg_disp_cache = None
        self._seg_slices_cache = None
        self._samples_item = None
        self._sel_key = None
        self.base_seg_qimg = None
        self.raw_display_rgb = None
        self.raw_pixmap = None
        self.classified_qimg = None
        self.classified_image_data = None
        self.raw_classified_raster_ints = None
        self.sieved_classified_raster_ints = None
        self.boundary_overlay_qimg = None
        self.hillshade_qimg = None
        self.current_view_mode = "segmented"
        self.sample_class_map = None

        self.selection_mode = False
        self.selected_segments = set()
        self.locked_selection = set()
        self.current_seed_segments = set()
        self.selection_undo_stack = []
        self._selection_drag_active = False

        self.acc_checked_point_ids = set()
        self.acc_nav_index = -1
        self.acc_point_move_id = None
        self.iterations = []

        # Load/View/Segment tab widgets
        self.path_edit.clear()
        self.dsm_path_edit.clear()
        self._configure_resample_controls_for_new_source()  # fresh engine -> hides Target Resolution, disables Apply
        self.reset_sliders()
        self._seg_size_drives = True
        self._set_segment_size_quietly(self.DEFAULT_SEGMENT_SIZE_M2)
        self._set_segment_count_quietly(5000)
        self.seg_plan_label.setText("")
        self.compactness_dspinbox.setValue(0.15)
        self.sigma_dspinbox.setValue(2.5)
        (self.gpu_radio if cuda_available() else self.cpu_radio).setChecked(True)
        self.show_boundaries_check.setChecked(False)
        self.show_boundaries_check.setEnabled(False)
        self.show_elevation_check.setChecked(False)
        self.show_elevation_check.setEnabled(False)
        self.elevation_opacity_slider.setEnabled(False)
        self.viewer.set_image(QImage())

        # Classification tab widgets
        self.update_dsm_status_label()
        self.class_list_widget.clear()
        self.selection_count_label.setText("Selected: 0 segments")
        self.selection_mode_btn.setChecked(False)
        self.selection_mode_btn.setText("Selection Mode: OFF")
        self.class_viewer.set_interaction_mode("paint")
        self.gen_slider.setValue(0)
        self.seg_view_btn.setChecked(True)
        self.cls_view_btn.setChecked(False)
        self.sync_classified_button_label()
        self._sync_blend_availability()
        self.class_viewer.set_image(QImage())

        # Accuracy Assessment tab widgets
        self.acc_csv_path_edit.clear()
        self.acc_csv_summary_label.setText("")
        self.acc_supp_expert_check.setChecked(False)
        self.acc_gt_expert_radio.setChecked(True)
        self.acc_sample_size_spinbox.setValue(200)
        self.use_deep_check.setChecked(False)
        self.classifier_features = None
        self.acc_design_combo.setCurrentIndex(0)
        self.acc_min_per_class_spinbox.setValue(50)
        self.acc_filter_combo.setCurrentIndex(0)
        self.acc_viewer.set_image(QImage())
        self.sync_accuracy_gt_controls()
        self.refresh_accuracy_ui()
        self.refresh_iteration_table()

        self._discover_reset()
        self.tabs.setCurrentIndex(0)
        print("Interface reset. Ready for a new orthomosaic.")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Set at the QApplication level too, not just on the main window -- on Windows
    # this is what determines the taskbar icon during startup (before the main
    # window's own icon would otherwise take over) and is what any top-level dialogs
    # inherit by default if they don't set their own.
    app_icon_path = get_resource_path(app_info.ICON_FILE)
    if os.path.exists(app_icon_path):
        app.setWindowIcon(QIcon(app_icon_path))

    window = ObiaApp()
    window.show()
    sys.exit(app.exec())