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

from PySide6.QtCore import QObject, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                               QVBoxLayout, QWidget)

import app_info as A
import obia_odm as odm
from obia_ui import CollapsibleSection

POLL_S = 5
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
                    return self._end_cancelled(already=True)
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
            c.remove(job["project"], job["task"])
            self.finished.emit(ortho, dsm)
        except Exception as e:
            self.failed.emit(str(e))

    def _end_cancelled(self, already=False):
        try:
            if not already:
                self.client.cancel(self.job["project"], self.job["task"])
            self.client.remove(self.job["project"], self.job["task"])
        except odm.WebODMError:
            pass
        self.cancelled.emit()


class OdmWindow(QDialog):
    """ The floating WebODM window. """
    status_text = Signal(str)            # for the main window's status bar ("" = no job)
    load_requested = Signal(str, str)    # ortho, dsm

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Orthomosaic from Drone Photos (WebODM)")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setModal(False)
        self.resize(560, 640)
        self.thread = self.worker = None
        self.summary = None
        self.started_at = None
        s = _settings()

        lay = QVBoxLayout(self)
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
        locked = QLabel("Settings: " + ", ".join(
            f"{k} {str(v).lower() if isinstance(v, bool) else v}" for k, v in odm.BASE_OPTIONS.items()))
        locked.setWordWrap(True)
        locked.setStyleSheet("color: #76817b; font-size: 11px;")
        f2.addRow("", locked)
        lay.addWidget(box)

        # Run
        btns = QHBoxLayout()
        self.start_btn = QPushButton("Start Processing")
        self.start_btn.clicked.connect(self.start)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        btns.addWidget(self.start_btn)
        btns.addWidget(self.cancel_btn)
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
        self.log_section = CollapsibleSection("WebODM log", self.log_view, start_collapsed=True)
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
        if not self.out_edit.text():
            self.out_edit.setText(folder.rstrip("/\\") + "_webodm")

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
               "options": odm.build_options(self.quality_combo.currentText(), gsd),
               "photos": photos, "started": time.time()}
        self.log_view.clear()
        self._run(client, job)

    def resume(self):
        """ Reattach to the job a previous session left running. """
        job = saved_job()
        if not job or self.thread is not None:
            return
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

    def _reconnect(self):
        self.start_btn.setText("Start Processing")
        self.start_btn.clicked.disconnect()
        self.start_btn.clicked.connect(self.start)
        self.resume()

    def _run(self, client, job):
        self.started_at = job.get("started") or time.time()
        self.thread = QThread()
        self.worker = OdmJobWorker(client, job)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.job_saved.connect(_save_job)
        self.worker.upload_progress.connect(lambda d, n: self._show("Uploading photos", d / max(n, 1),
                                                                    f"{d}/{n}"))
        self.worker.status.connect(lambda st, p, code: self._show(st, p))
        self.worker.log_lines.connect(lambda lines: self.log_view.appendPlainText("\n".join(lines)))
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
