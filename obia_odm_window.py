"""Import > Orthomosaic from Drone Photos (WebODM): a floating window that runs one WebODM
job at a time through obia_odm, shows its progress and console, and hands the finished
orthomosaic and DSM to the main window.

The window is non-modal and closing it does not stop the job. The job's ids are kept in
QSettings, so Lite Classify can be closed during a long run and picks the job up again on
the next start (WebODM keeps processing either way).
"""
import json
import os
import time
import urllib.parse

from PySide6.QtCore import QObject, QPointF, QRectF, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGraphicsItem, QGraphicsScene, QGraphicsView,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QRadioButton,
                               QLayout,
                               QVBoxLayout, QWidget)

import app_info as A
import obia_odm as odm
from obia_ui import CollapsibleSection

POLL_S = 5
REMOVE_WAIT_S = 120
ORTHO_NAME, DSM_NAME = "odm_orthophoto.tif", "odm_dsm.tif"


def _settings():
    return QSettings(A.APP_AUTHOR, A.APP_NAME)


def saved_job():
    """ The job left running by a previous session, or None. """
    raw = _settings().value("webodm/job", "")
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return None


def _save_job(job):
    _settings().setValue("webodm/job", json.dumps(job) if job else "")


class OdmJobWorker(QObject):
    """ Runs (or reattaches to) one WebODM task to the end, off the UI thread.
    job: {"project", "task" (None for a new job), "out_dir", "gsd_cm", "options",
    "photos" (new jobs only), "name"}. """
    job_saved = Signal(dict)              # ids known: persist so a restart can reattach
    upload_progress = Signal(int, int)
    status = Signal(str, float, int)      # stage, progress 0-1, WebODM status code
    log_lines = Signal(list)
    finished = Signal(str, str)           # ortho path, dsm path
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, client, job):
        super().__init__()
        self.client, self.job = client, dict(job)
        self._cancel = False
        self._detach = False              # app closing: stop watching, leave the job running

    def request_cancel(self):
        self._cancel = True

    def detach(self):
        self._detach = True

    def _stopped(self):
        return self._cancel or self._detach

    def _sleep(self, s):
        end = time.time() + s
        while time.time() < end and not self._stopped():
            time.sleep(0.2)

    def run(self):
        c, job = self.client, self.job
        try:
            if job.get("task") is None:
                if not c.online_nodes():
                    raise odm.WebODMError("WebODM has no processing node online. Open WebODM and "
                                          "check Processing Nodes.")
                job["project"] = c.ensure_project()
                t = c.create_task(job["project"], job["name"], job["options"])
                job["task"] = t["id"]
                bad = odm.options_match(job["options"], c.task(job["project"], job["task"]).get("options"))
                if bad:
                    c.remove(job["project"], job["task"])
                    raise odm.WebODMError("WebODM changed the processing settings:\n" + "\n".join(bad))
                saved = {k: v for k, v in job.items() if k != "photos"}
                self.job_saved.emit(saved)
                self.status.emit("Uploading photos", 0.0, 0)
                ok = c.upload(job["project"], job["task"], job["photos"],
                              progress=lambda d, n: self.upload_progress.emit(d, n),
                              cancelled=self._stopped)
                if not ok:
                    if self._cancel:
                        return self._end_cancelled()
                    # App closing mid-upload: an uncommitted task can't be resumed, so drop it.
                    c.remove(job["project"], job["task"])
                    _save_job(None)
                    return
                c.commit(job["project"], job["task"])

            line, stage, capped = job.get("line", 0), None, []
            while True:
                if self._detach:
                    return
                if self._cancel:
                    return self._end_cancelled()
                info = c.task(job["project"], job["task"])
                lines, line = c.output(job["project"], job["task"], line)
                if lines:
                    stage = odm.stage_from_lines(lines, stage)
                    capped += [ln for ln in lines if "Maximum resolution set to" in ln]
                    self.log_lines.emit(lines)
                st = info.get("status")
                self.status.emit(stage or odm.STATUS_NAMES.get(st, "Waiting"),
                                 float(info.get("running_progress") or 0), st or 0)
                if st == odm.COMPLETED:
                    break
                if st == odm.FAILED:
                    raise odm.WebODMError(f"WebODM reported a failure: {info.get('last_error') or 'see the log'}")
                if st == odm.CANCELED:
                    return self._end_cancelled()
                self._sleep(POLL_S)

            os.makedirs(job["out_dir"], exist_ok=True)
            ortho, dsm = (os.path.join(job["out_dir"], n) for n in (ORTHO_NAME, DSM_NAME))
            for asset, dest, label in (("orthophoto.tif", ortho, "Downloading orthomosaic"),
                                       ("dsm.tif", dsm, "Downloading DSM")):
                self.status.emit(label, 0.0, odm.COMPLETED)
                done = c.download(job["project"], job["task"], asset, dest,
                                  progress=lambda g, n, l=label: self.status.emit(l, g / n if n else 0.0, odm.COMPLETED),
                                  cancelled=self._stopped)
                if not done:
                    return self._end_cancelled() if self._cancel else None
            problems = odm.check_outputs(ortho, dsm, job.get("gsd_cm"), capped)
            if problems:
                # Keep the task in WebODM so nothing is lost while this is looked into.
                raise odm.WebODMError("The downloaded files did not pass the checks (the task is kept "
                                      "in WebODM):\n" + "\n".join(problems))
            try:
                c.remove(job["project"], job["task"], wait_s=REMOVE_WAIT_S)
            except odm.WebODMError as e:
                self.log_lines.emit([str(e)])           # the results are safe; only cleanup lags
            self.finished.emit(ortho, dsm)
        except Exception as e:
            self.failed.emit(str(e))

    def _end_cancelled(self):
        try:
            self.client.remove(self.job["project"], self.job["task"], wait_s=REMOVE_WAIT_S)
        except odm.WebODMError as e:
            return self.failed.emit(f"Cancelled, but {e}")
        self.cancelled.emit()


class PhotoMapView(QGraphicsView):
    """ Photo positions (dots, joined in capture order) on a plain background, or over the
    loaded orthomosaic, with rectangle or polygon drawing of an area of interest.
    Map coordinates are metres east/north; the scene flips y. """
    aoi_changed = Signal(list)            # [(x, y), ...] map coordinates; [] = none

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing)
        self.setMinimumHeight(260)
        self.setBackgroundBrush(QColor("#f3f4f1"))
        self.setDragMode(QGraphicsView.NoDrag)
        self.viewport().setMouseTracking(True)      # polygon edge follows the cursor
        # No scroll bars: their appearing on zoom-in resized the view, which refitted it.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._pan = None
        self.mode = "rect"
        self._dots, self._aoi_item, self._draft, self._corners = [], None, [], []
        self._press = None
        self._r = 1.0
        self._fit_scale = None

    def set_photos(self, xy, underlay=None):
        """ xy: [(x, y)]; underlay: (QImage, (left, bottom, right, top)) or None. """
        sc = self.scene()
        sc.clear()
        self._dots, self._aoi_item, self._draft, self._corners = [], None, [], []
        if not xy:
            return
        xs, ys = [p[0] for p in xy], [p[1] for p in xy]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 10.0)
        self._r = span / 150
        if underlay is not None:
            img, (l, b, r, t) = underlay
            item = sc.addPixmap(QPixmap.fromImage(img))
            item.setOffset(0, 0)
            item.setScale((r - l) / img.width())
            item.setPos(l, -t)
            item.setOpacity(0.85)
        path = QPainterPath(QPointF(xs[0], -ys[0]))
        for x, y in xy[1:]:
            path.lineTo(x, -y)
        sc.addPath(path, QPen(QColor("#9aa39d"), 0))
        for x, y in xy:
            d = sc.addEllipse(x - self._r, -y - self._r, 2 * self._r, 2 * self._r, QPen(Qt.NoPen),
                              QBrush(QColor("#f65244")))
            self._dots.append(d)
        pad = span * 0.08
        self.setSceneRect(QRectF(min(xs) - pad, -max(ys) - pad, max(xs) - min(xs) + 2 * pad,
                                 max(ys) - min(ys) + 2 * pad))
        QTimer.singleShot(0, self._fit)

    def _fit(self):
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self._fit_scale = self.transform().m11()

    def set_selected(self, mask):
        on, off = QBrush(QColor("#f65244")), QBrush(QColor("#b9c0bb"))
        for d, m in zip(self._dots, mask):
            d.setBrush(on if m else off)

    def clear_aoi(self):
        self._draft = []
        self._show_aoi([])
        self.aoi_changed.emit([])

    def _show_aoi(self, pts, closed=True, corners=None):
        if self._aoi_item is not None:
            self.scene().removeItem(self._aoi_item)
            self._aoi_item = None
        for c in self._corners:
            self.scene().removeItem(c)
        self._corners = []
        pen = QPen(QColor("#1f6fb2"), 1.5)
        pen.setCosmetic(True)
        for x, y in pts[:corners]:         # corner markers, a constant size on screen
            c = self.scene().addRect(-4, -4, 8, 8, pen, QBrush(QColor("white")))
            c.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            c.setPos(x, -y)
            c.setZValue(2)
            self._corners.append(c)
        if len(pts) >= 2:
            poly = QPolygonF([QPointF(x, -y) for x, y in pts])
            pen = QPen(QColor("#1f6fb2"), 0)
            pen.setWidthF(2)
            pen.setCosmetic(True)
            if closed:
                self._aoi_item = self.scene().addPolygon(poly, pen, QBrush(QColor(31, 111, 178, 40)))
            else:
                path = QPainterPath()
                path.addPolygon(poly)
                self._aoi_item = self.scene().addPath(path, pen)

    def _map_pt(self, event):
        p = self.mapToScene(event.position().toPoint())
        return (p.x(), -p.y())

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.clear_aoi()
            return
        if event.button() == Qt.MiddleButton:
            self._pan = event.position()
            return
        if event.button() != Qt.LeftButton or not self._dots:
            return super().mousePressEvent(event)
        pt = self._map_pt(event)
        if self.mode == "rect":
            self._press = pt
        else:
            self._draft.append(pt)
            self._show_aoi(self._draft, closed=False)

    def mouseMoveEvent(self, event):
        if self._pan is not None:                 # middle-drag pans
            d = event.position() - self._pan
            self._pan = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        if self.mode == "rect" and self._press is not None:
            (x0, y0), (x1, y1) = self._press, self._map_pt(event)
            self._show_aoi([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        elif self.mode == "poly" and self._draft:
            self._show_aoi(self._draft + [self._map_pt(event)], closed=False, corners=len(self._draft))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._pan = None
            return
        if self.mode == "rect" and self._press is not None and event.button() == Qt.LeftButton:
            (x0, y0), (x1, y1) = self._press, self._map_pt(event)
            self._press = None
            if abs(x1 - x0) > self._r and abs(y1 - y0) > self._r:
                pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                self._show_aoi(pts)
                self.aoi_changed.emit(pts)
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.mode == "poly" and len(self._draft) >= 3:
            pts, self._draft = self._draft, []
            self._show_aoi(pts)
            self.aoi_changed.emit(pts)

    def wheelEvent(self, event):
        if not self._fit_scale:
            return
        f = 1.25 if event.angleDelta().y() > 0 else 0.8
        # Zoom out no further than the whole flight, in no closer than 40x.
        cur = self.transform().m11()
        f = min(max(f, self._fit_scale / cur), 40 * self._fit_scale / cur)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale(f, f)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._dots and (self._fit_scale is None or self.transform().m11() <= self._fit_scale * 1.001):
            self._fit()


class OdmWindow(QDialog):
    """ The floating WebODM window. """
    status_text = Signal(str)            # for the main window's status bar ("" = no job)
    load_requested = Signal(str, str)    # ortho, dsm

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Orthomosaic from Drone Photos (WebODM)")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setModal(False)
        self.thread = self.worker = None
        self.summary = None
        self.started_at = None
        s = _settings()

        outer = QHBoxLayout(self)
        outer.setSizeConstraint(QLayout.SetFixedSize)   # grows and shrinks with the map panel
        left = QWidget()
        left.setFixedWidth(540)
        lay = QVBoxLayout(left)
        lay.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(left)
        intro = QLabel("Processes a folder of drone photos with your WebODM installation and loads the "
                       "orthomosaic and DSM into Lite Classify. WebODM must be installed and running.")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        # Connection
        conn = QWidget()
        form = QFormLayout(conn)
        url_row = QHBoxLayout()
        self.url_edit = QLineEdit(s.value("webodm/url", ""))
        self.url_edit.setPlaceholderText("http://127.0.0.1:29800")
        find_btn = QPushButton("Find")
        find_btn.setToolTip("Look for WebODM on this PC")
        find_btn.clicked.connect(self._find_webodm)
        url_row.addWidget(self.url_edit)
        url_row.addWidget(find_btn)
        form.addRow("WebODM address:", url_row)
        self.user_edit = QLineEdit(s.value("webodm/user", ""))
        form.addRow("Username:", self.user_edit)
        self.pass_edit = QLineEdit(odm.unprotect(s.value("webodm/password", "")) or "")
        self.pass_edit.setEchoMode(QLineEdit.Password)
        form.addRow("Password:", self.pass_edit)
        self.remember_check = QCheckBox("Remember me (stored encrypted for your Windows account)")
        self.remember_check.setChecked(s.value("webodm/remember", "false") == "true")
        self.remember_check.setEnabled(os.name == "nt")
        form.addRow("", self.remember_check)
        has_login = bool(self.user_edit.text() and self.pass_edit.text())
        lay.addWidget(CollapsibleSection("Connection", conn, start_collapsed=has_login))

        # Photos and output
        box = QGroupBox("Photos")
        f2 = QFormLayout(box)
        photo_row = QHBoxLayout()
        self.photos_edit = QLineEdit()
        self.photos_edit.setPlaceholderText("Folder of drone photos…")
        self.photos_edit.editingFinished.connect(self._summarize)
        b = QPushButton("Browse")
        b.clicked.connect(self._browse_photos)
        photo_row.addWidget(self.photos_edit)
        photo_row.addWidget(b)
        f2.addRow("Photo folder:", photo_row)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        f2.addRow("", self.summary_label)

        # The area map opens beside the form, so the window grows sideways, not off the screen.
        self.area_panel = area = QGroupBox("Area to process")
        al = QVBoxLayout(area)
        tools = QHBoxLayout()
        self.rect_radio = QRadioButton("Rectangle")
        self.poly_radio = QRadioButton("Polygon")
        self.rect_radio.setChecked(True)
        self.rect_radio.toggled.connect(lambda on: setattr(self.map_view, "mode", "rect" if on else "poly"))
        all_btn = QPushButton("Use All Photos")
        tools.addWidget(QLabel("Draw:"))
        tools.addWidget(self.rect_radio)
        tools.addWidget(self.poly_radio)
        tools.addStretch(1)
        tools.addWidget(all_btn)
        al.addLayout(tools)
        self.map_view = PhotoMapView()
        self.map_view.setMinimumSize(600, 520)
        self.map_view.aoi_changed.connect(self._on_aoi)
        all_btn.clicked.connect(self.map_view.clear_aoi)
        al.addWidget(self.map_view)
        tip = QLabel("Drag a rectangle, or click polygon corners and double-click to finish. Photos taken "
                     "inside the area are processed and the results are cropped to it; its edges see fewer "
                     "photos, so draw a little beyond what you need. Scroll to zoom, middle-drag to pan; "
                     "right-click clears.")
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #76817b; font-size: 11px;")
        al.addWidget(tip)
        area.hide()
        outer.addWidget(area, 1)
        area_row = QHBoxLayout()
        area_row.addWidget(QLabel("Process only part of the flight (optional)"))
        area_row.addStretch(1)
        self.area_btn = QPushButton("Choose Area \u25b8")
        self.area_btn.setCheckable(True)
        self.area_btn.setToolTip("Open a map of the photos beside this form and draw the area to process")
        self.area_btn.toggled.connect(self._toggle_area)
        area_row.addWidget(self.area_btn)
        f2.addRow(area_row)
        self.selection_label = QLabel("")
        self.selection_label.setWordWrap(True)
        f2.addRow("", self.selection_label)
        self.positions, self.epsg, self.aoi, self.selected = [], None, [], []
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(list(odm.QUALITY_PRESETS))
        self.quality_combo.setToolTip("Standard aligns the photos at half resolution; Maximum at full "
                                      "resolution, which is much slower for little geometric gain.")
        f2.addRow("Quality:", self.quality_combo)
        self.res_spin = QDoubleSpinBox()
        self.res_spin.setRange(0.5, 50.0)
        self.res_spin.setDecimals(2)
        self.res_spin.setSuffix(" cm/px")
        self.res_spin.setValue(5.0)
        self.res_spin.setToolTip("Orthomosaic resolution; the DSM is made at 4x this. Estimated from "
                                 "the photos (flight height and camera).")
        f2.addRow("Resolution:", self.res_spin)
        out_row = QHBoxLayout()
        self.out_edit = QLineEdit()
        ob = QPushButton("Browse")
        ob.clicked.connect(self._browse_out)
        out_row.addWidget(self.out_edit)
        out_row.addWidget(ob)
        f2.addRow("Save results to:", out_row)
        lay.addWidget(box)

        # Run
        btns = QHBoxLayout()
        self.start_btn = QPushButton("Start Processing")
        self.start_btn.clicked.connect(self.start)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        self.clean_btn = QPushButton("Clean Up WebODM\u2026")
        self.clean_btn.setToolTip("Remove leftover Lite Classify tasks and stale upload files from WebODM")
        self.clean_btn.clicked.connect(self.clean_up)
        btns.addWidget(self.start_btn)
        btns.addWidget(self.cancel_btn)
        btns.addWidget(self.clean_btn)
        lay.addLayout(btns)

        self.stage_label = QLabel("")
        lay.addWidget(self.stage_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        lay.addWidget(self.progress)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        self.log_view.setFont(QFont("Consolas", 8))
        self.log_view.setMinimumHeight(160)
        self.log_section = CollapsibleSection("WebODM log", self.log_view, start_collapsed=False)
        lay.addWidget(self.log_section, 1)

        self.clock = QTimer(self)
        self.clock.timeout.connect(self._tick)
        self._stage, self._frac = "", 0.0

        if saved_job():
            QTimer.singleShot(0, self.resume)

    # --- helpers ---
    def _find_webodm(self):
        url = odm.find_local_webodm()
        if url:
            self.url_edit.setText(url)
        else:
            QMessageBox.information(self, "WebODM", "WebODM was not found on this PC. Start WebODM, "
                                    "or type its address.")

    def _browse_photos(self):
        d = QFileDialog.getExistingDirectory(self, "Folder of drone photos", self.photos_edit.text())
        if d:
            self.photos_edit.setText(d)
            self._summarize()

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Save results to", self.out_edit.text())
        if d:
            self.out_edit.setText(d)

    def _summarize(self):
        folder = self.photos_edit.text().strip()
        if not os.path.isdir(folder):
            self.summary = None
            self.summary_label.setText("")
            return
        s = self.summary = odm.photo_summary(folder)
        if not s["count"]:
            self.summary_label.setText("No photos found in this folder.")
            return
        parts = [f"{s['count']} photos ({s['bytes'] / 1e9:.1f} GB)"]
        if s["camera"]:
            parts.append(s["camera"])
        if s["width"]:
            parts.append(f"{s['width']}x{s['height']}")
        if s["altitude_m"]:
            parts.append(f"~{s['altitude_m']:.0f} m above take-off")
        if s["gsd_cm"]:
            parts.append(f"GSD ~{s['gsd_cm']:.2f} cm")
            self.res_spin.setValue(round(s["gsd_cm"], 2))
        free = odm.free_disk_bytes(os.path.dirname(folder.rstrip("/\\")) or folder)
        if free is not None and free < 3 * s["bytes"]:
            parts.append(f"warning: only {free / 1e9:.1f} GB free on this drive")
        self.summary_label.setText(", ".join(parts) + ".")
        self._load_positions()
        if not self.out_edit.text():
            self.out_edit.setText(folder.rstrip("/\\") + "_webodm")

    def _load_positions(self):
        """ Photo GPS positions for the area map (projected to UTM, metres). """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            pos = odm.photo_positions(odm.list_images(self.photos_edit.text().strip()))
        finally:
            QApplication.restoreOverrideCursor()
        self.positions, self.aoi = pos, []
        if not pos:
            self.epsg, self.selected = None, []
            self.map_view.set_photos([])
            self.selection_label.setText("These photos have no GPS positions; all of them will be used.")
            return
        self.epsg = odm.utm_epsg(pos[0]["lon"], pos[0]["lat"])
        xs, ys = odm.project([p["lon"] for p in pos], [p["lat"] for p in pos], self.epsg)
        self.xy = list(zip(xs, ys))
        self.map_view.set_photos(self.xy, self._underlay())
        self._update_selection()

    def _underlay(self):
        """ The orthomosaic loaded in the main window, if it is in the same UTM zone. """
        e = getattr(self.parent(), "engine", None)
        if e is None or e.ortho_data is None or not e.ortho_meta or self.epsg is None:
            return None
        try:
            if e.ortho_meta["crs"].to_epsg() != self.epsg:
                return None
            import numpy as np
            b, h, w = e.ortho_data.shape
            step = max(1, max(h, w) // 1500)
            rgb = np.ascontiguousarray(np.transpose(e.ortho_data[:3, ::step, ::step], (1, 2, 0)).astype(np.uint8))
            img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format_RGB888).copy()
            tr = e.ortho_meta["transform"]
            return img, (tr.c, tr.f + tr.e * h, tr.c + tr.a * w, tr.f)
        except Exception:
            return None

    def _toggle_area(self, on):
        self.area_btn.setText("Hide Map \u25c2" if on else "Choose Area \u25b8")
        self.area_panel.setVisible(on)        # the window resizes itself (fixed-size layout)

    def _on_aoi(self, pts):
        self.aoi = pts
        self._update_selection()

    def _update_selection(self):
        if not self.positions:
            return
        self.selected = (odm.select_photos(self.xy, self.aoi) if self.aoi
                         else [True] * len(self.positions))
        self.map_view.set_selected(self.selected)
        n = sum(self.selected)
        size = sum(os.path.getsize(p["path"]) for p, m in zip(self.positions, self.selected) if m)
        what = ("inside the drawn area" if self.aoi else
                "the whole flight")
        self.selection_label.setText(f"{n} of {len(self.positions)} photos selected ({size / 1e9:.1f} GB), "
                                     f"{what}.")

    def clean_up(self):
        """ Removes what Lite Classify jobs can leave in WebODM: tasks in the Lite Classify
        project (except the running job's) and, for a WebODM on this PC, upload temp files
        it did not delete. Lists everything and asks first. """
        try:
            c = self._client()
        except odm.WebODMError as e:
            QMessageBox.warning(self, "WebODM", str(e))
            return
        running = (self.worker.job if self.worker is not None else saved_job()) or {}
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            pid = c.find_project()
            tasks = [t for t in (c.tasks(pid) if pid else []) if t.get("id") != running.get("task")]
            local = urllib.parse.urlparse(c.url).hostname in ("127.0.0.1", "localhost")
            app_dir = odm.local_webodm_app_dir() if local else None
            stale = odm.stale_upload_files(app_dir) if app_dir else []
        except odm.WebODMError as e:
            QMessageBox.warning(self, "WebODM", str(e))
            return
        finally:
            QApplication.restoreOverrideCursor()
        if not tasks and not stale:
            QMessageBox.information(self, "Clean Up WebODM", "Nothing to clean up."
                                    + ("" if app_dir or not local else "\n(WebODM's own folder was "
                                       "not found, so its upload files were not checked.)"))
            return
        lines = []
        if tasks:
            lines.append(f"Remove {len(tasks)} task(s) from the \"{odm.PROJECT_NAME}\" project in WebODM:")
            for t in tasks[:12]:
                size = f", {t['size'] / 1024:.1f} GB" if t.get("size") else ""
                lines.append(f"  • {t.get('name') or t['id']} "
                             f"({odm.STATUS_NAMES.get(t.get('status'), 'Waiting')}{size})")
            if len(tasks) > 12:
                lines.append(f"  • … and {len(tasks) - 12} more")
        if stale:
            gb = sum(s for _, s in stale) / 1e9
            lines.append(f"Delete {len(stale)} leftover upload file(s), {gb:.2f} GB, from "
                         f"{os.path.join(app_dir, 'data', 'media', 'tmp')} and NodeODX's tmp folder.")
        if QMessageBox.question(self, "Clean Up WebODM", "\n".join(lines) + "\n\nThis cannot be undone.") \
                != QMessageBox.Yes:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for t in tasks:
                c.remove(pid, t["id"])
            left, end = list(tasks), time.time() + 120
            while left and time.time() < end:
                QApplication.processEvents()
                time.sleep(0.5)
                left = [t for t in left if c.exists(pid, t["id"])]
            freed, failed = odm.delete_paths(stale)
        except odm.WebODMError as e:
            QMessageBox.warning(self, "WebODM", str(e))
            return
        finally:
            QApplication.restoreOverrideCursor()
        msg = []
        if tasks:
            msg.append(f"Removed {len(tasks) - len(left)} of {len(tasks)} task(s)."
                       + (" WebODM is still deleting the rest." if left else ""))
        if stale:
            msg.append(f"Deleted {freed / 1e9:.2f} GB of upload files."
                       + (f" {len(failed)} could not be deleted (in use)." if failed else ""))
        QMessageBox.information(self, "Clean Up WebODM", "\n".join(msg))

    def _client(self):
        url, user, pw = self.url_edit.text().strip(), self.user_edit.text().strip(), self.pass_edit.text()
        if not url:
            url = odm.find_local_webodm() or ""
            self.url_edit.setText(url)
        if not (url and user and pw):
            raise odm.AuthError("Enter the WebODM address, username and password under Connection.")
        s = _settings()
        s.setValue("webodm/url", url)
        s.setValue("webodm/user", user)
        s.setValue("webodm/remember", "true" if self.remember_check.isChecked() else "false")
        s.setValue("webodm/password", (odm.protect(pw) or "") if self.remember_check.isChecked() else "")
        c = odm.WebODMClient(url, user, pw)
        c.login()
        return c

    # --- run ---
    def start(self):
        if self.thread is not None:
            return
        folder = self.photos_edit.text().strip()
        photos = odm.list_images(folder) if os.path.isdir(folder) else []
        boundary = None
        if self.aoi and self.positions:
            photos = [p["path"] for p, m in zip(self.positions, self.selected) if m]
            boundary = odm.boundary_geojson(self.aoi, self.epsg)
        if len(photos) < 2:
            QMessageBox.warning(self, "WebODM", "Choose a folder with at least two drone photos.")
            return
        out_dir = self.out_edit.text().strip() or folder.rstrip("/\\") + "_webodm"
        if any(os.path.exists(os.path.join(out_dir, n)) for n in (ORTHO_NAME, DSM_NAME)):
            if QMessageBox.question(self, "WebODM", f"{out_dir} already has results. Replace them when "
                                    "this job finishes?") != QMessageBox.Yes:
                return
        try:
            client = self._client()
        except odm.WebODMError as e:
            QMessageBox.warning(self, "WebODM", str(e))
            return
        gsd = self.res_spin.value()
        name = os.path.basename(folder.rstrip("/\\"))
        job = {"project": None, "task": None, "out_dir": out_dir, "gsd_cm": gsd,
               "name": f"{name} (Lite Classify)",
               "options": odm.build_options(self.quality_combo.currentText(), gsd, boundary),
               "photos": photos, "started": time.time(),
               "folder": folder, "aoi": [list(p) for p in self.aoi]}
        self.log_view.clear()
        self.log_view.appendPlainText(f"{len(photos)} photos, settings sent to WebODM:\n" + " ".join(
            f"--{o['name']} {o['value'] if o['name'] != 'boundary' else '<drawn area>'}"
            for o in job["options"]))
        self._run(client, job)

    def resume(self):
        """ Reattach to the job a previous session left running. """
        job = saved_job()
        if not job or self.thread is not None:
            return
        self._show_job_photos(job)
        try:
            client = self._client()
        except odm.WebODMError as e:
            self.stage_label.setText(f"A WebODM job from an earlier session is waiting: {e}")
            self.status_text.emit("WebODM job waiting: log in")
            self.start_btn.setText("Reconnect")
            self.start_btn.clicked.disconnect()
            self.start_btn.clicked.connect(self._reconnect)
            return
        self.out_edit.setText(job["out_dir"])
        self._run(client, job)

    def _show_job_photos(self, job):
        """ Fills the photo folder and drawn area of a resumed job, so its map shows. """
        folder = job.get("folder")
        if not folder or not os.path.isdir(folder) or self.photos_edit.text().strip() == folder:
            return
        self.photos_edit.setText(folder)
        self._summarize()
        if job.get("gsd_cm"):
            self.res_spin.setValue(job["gsd_cm"])
        aoi = [tuple(p) for p in job.get("aoi") or []]
        if aoi and self.positions:
            self.map_view._show_aoi(aoi)
            self._on_aoi(aoi)

    def _reconnect(self):
        self.start_btn.setText("Start Processing")
        self.start_btn.clicked.disconnect()
        self.start_btn.clicked.connect(self.start)
        self.resume()

    def _on_job_saved(self, job):
        _save_job(job)

    def _on_upload_progress(self, done, total):
        self._show("Uploading photos", done / max(total, 1), f"{done}/{total}")

    def _on_status(self, stage, frac, code):
        self._show(stage, frac)

    def _on_log_lines(self, lines):
        self.log_view.appendPlainText("\n".join(lines))

    def _run(self, client, job):
        self.started_at = job.get("started") or time.time()
        self.thread = QThread()
        self.worker = OdmJobWorker(client, job)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        # Bound methods, not lambdas: Qt queues those onto the UI thread. A lambda runs in
        # the worker's thread and touching widgets from there crashes the app.
        self.worker.job_saved.connect(self._on_job_saved)
        self.worker.upload_progress.connect(self._on_upload_progress)
        self.worker.status.connect(self._on_status)
        self.worker.log_lines.connect(self._on_log_lines)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.cancelled.connect(self._on_cancelled)
        for sig in (self.worker.finished, self.worker.failed, self.worker.cancelled):
            sig.connect(self.thread.quit)
        self.thread.finished.connect(self._cleanup_thread)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.setText("Cancel")
        self.clock.start(1000)
        self._show("Connecting", 0.0)
        self.thread.start()

    def _show(self, stage, frac, extra=""):
        self._stage, self._frac = stage, frac
        self.progress.setValue(int(1000 * max(0.0, min(1.0, frac))))
        self._tick(extra)

    def _tick(self, extra=""):
        if self.started_at is None:
            return
        el = int(time.time() - self.started_at)
        clock = f"{el // 3600}:{el // 60 % 60:02d}:{el % 60:02d}"
        pct = f"{100 * self._frac:.0f}%"
        self.stage_label.setText(f"{self._stage}  {extra or pct}   ·   elapsed {clock}")
        self.status_text.emit(f"WebODM {pct} · {self._stage}")

    def cancel(self):
        if self.worker is None:
            return
        el = (time.time() - self.started_at) / 3600 if self.started_at else 0
        if QMessageBox.question(self, "Cancel WebODM job",
                                f"Stop this job (running {el:.1f} h) and remove it from WebODM?") != QMessageBox.Yes:
            return
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("Cancelling…")
        self.worker.request_cancel()

    def _end(self, text):
        self.clock.stop()
        self.started_at = None
        self.stage_label.setText(text)
        self.status_text.emit("")
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("Cancel")

    def _on_finished(self, ortho, dsm):
        _save_job(None)
        self.progress.setValue(1000)
        self._end(f"Done: {os.path.dirname(ortho)}")
        self.show()
        self.raise_()
        if QMessageBox.question(self, "WebODM", "The orthomosaic and DSM are ready.\n\nLoad them into "
                                "Load, View & Segment now?") == QMessageBox.Yes:
            self.load_requested.emit(ortho, dsm)

    def _on_failed(self, msg):
        _save_job(None)
        self._end("Failed.")
        self.log_view.appendPlainText("\n" + msg)
        self.log_section.toggle_btn.setChecked(True)
        self.log_section._on_toggled()
        self.show()
        QMessageBox.warning(self, "WebODM", msg)

    def _on_cancelled(self):
        _save_job(None)
        self.progress.setValue(0)
        self._end("Cancelled; the job was removed from WebODM.")

    def _cleanup_thread(self):
        self.thread.deleteLater()
        self.worker.deleteLater()
        self.thread = self.worker = None

    def detach(self):
        """ Lite Classify is closing: stop watching but leave the job running in WebODM
        (it is picked up again on the next start). """
        if self.worker is not None:
            self.worker.detach()
            self.thread.quit()
            self.thread.wait(3000)

    def is_running(self):
        return self.worker is not None
