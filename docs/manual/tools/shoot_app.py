"""Drives the real Lite Classify window through the sample workflow, then restores the
user's saved workspace (data/Ortho_sample_workspace), and saves screenshots for
the manual. Run from the repo root: python docs/manual/tools/shoot_app.py
A Lite Classify window appears on screen while it runs; do not click in it."""
import os, sys, time
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))

from PySide6.QtWidgets import QApplication, QListWidgetItem
from PySide6.QtGui import QColor
import obia_app
from obia_ui import CollapsibleSection
import json, re

OUT = os.path.join(os.path.dirname(__file__), "..", "img", "app")
HERE = os.path.dirname(__file__)


def log(*a):
    print(*a, file=sys.__stdout__, flush=True)


def pump(seconds=0.4):
    end = time.time() + seconds
    while time.time() < end:
        QApplication.processEvents()
        time.sleep(0.02)


def wait_for(cond, timeout=600):
    end = time.time() + timeout
    while not cond():
        if time.time() > end:
            raise TimeoutError
        QApplication.processEvents()
        time.sleep(0.05)
    pump(0.6)


def shot(w, name):
    os.makedirs(OUT, exist_ok=True)
    pump(0.5)
    png = os.path.join(OUT, name + ".png")
    w.grab().save(png)
    from PIL import Image   # WebP keeps UI text crisp at a tenth of the PNG size
    Image.open(png).convert("RGB").save(os.path.join(OUT, name + ".webp"), quality=88, method=6)
    os.remove(png)
    log("saved", name)


def main():
    app = QApplication(sys.argv)
    w = obia_app.ObiaApp()
    w.setGeometry(40, 40, 1600, 940)
    w.show()
    pump(1.0)
    shot(w, "01_startup")

    # Load ortho + DSM and resample to the workspace's 10 cm, as a user would
    w.path_edit.setText(os.path.abspath("data/Ortho_sample.tif"))
    w.load_ortho()
    w.dsm_path_edit.setText(os.path.abspath("data/DSM_sample.tif"))
    w.load_dsm()
    pump(1.0)
    shot(w, "02_loaded_native")
    w.resample_target_res_spinbox.setValue(10.0)
    pump(0.3)
    shot(w, "03_resample_set")
    w.apply_resample()
    pump(1.0)
    w.viewer.fit_view()
    shot(w, "04_resampled")

    sec = w.findChildren(CollapsibleSection)[0]
    sec.toggle_btn.setChecked(True); sec._on_toggled()
    pump(0.5)
    shot(w, "05_color_panel")
    sec.toggle_btn.setChecked(False); sec._on_toggled()

    w.show_elevation_check.setChecked(True)
    pump(0.8)
    shot(w, "06_elevation_overlay")
    w.show_elevation_check.setChecked(False)

    # A live segmentation run with the workspace's settings, for the timing and screenshots
    w.n_segments_spinbox.setValue(8000)
    w.compactness_dspinbox.setValue(0.15)
    w.sigma_dspinbox.setValue(2.5)
    w.gpu_radio.setChecked(True)
    w.tabs.setCurrentIndex(0)
    w.run_segmentation()
    pump(0.5)
    shot(w, "07_segmenting")
    wait_for(lambda: w.run_seg_btn.isEnabled())
    m = re.search(r"Compiled (\d+) polygons in ([\d.]+) seconds", w.console.toPlainText())
    json.dump({"polygons": int(m.group(1)), "seconds": float(m.group(2))},
              open(os.path.join(HERE, "..", "app_run.json"), "w"))
    log("segmentation", m.group(0))

    # Restore the user's saved session: segments, classes, samples, map, points
    w.load_workspace_session()
    pump(1.0)
    w.gen_spinbox.setValue(2000)
    w.apply_generalization_sieve()
    pump(0.5)

    w.tabs.setCurrentIndex(0)
    w.show_boundaries_check.setChecked(True)
    pump(0.5)
    w.viewer.center_on_pixmap_point(3728, 1440, zoom=0.6)
    shot(w, "09_boundaries_zoom")
    w.show_boundaries_check.setChecked(False)

    # Segmented view with the training samples
    w.tabs.setCurrentWidget(w.classification_tab)
    w.class_list_widget.setCurrentRow(0)
    w.on_class_selection_changed()
    w.seg_view_btn.setChecked(True); w.toggle_segmented_view()
    w.class_viewer.fit_view()
    shot(w, "10_training_samples")

    # Grow tool demo on the sports field
    w.selection_mode_btn.setChecked(True); w.toggle_selection_mode()
    seed = int(w.engine.segmented_image[1800, 2560])
    for tol, nm in ((0.5, "11_grow_050"), (1.0, "12_grow_100"), (2.0, "13_grow_200")):
        w.locked_selection = set(); w.current_seed_segments = {seed}
        w._set_grow_tolerance_silently(tol)
        w.recompute_grown_selection()
        w.class_viewer.center_on_pixmap_point(2480, 1840, zoom=0.27)
        shot(w, nm)
        log(nm, w.selection_count_label.text())
    w.clear_segment_selection()
    w._set_grow_tolerance_silently(1.0)
    w.selection_mode_btn.setChecked(False); w.toggle_selection_mode()

    # The workspace's classified map
    w.cls_view_btn.setChecked(True); w.toggle_classified_view()
    w.class_viewer.fit_view()
    shot(w, "14_classified")
    w.toggle_samples_btn.setChecked(False); w.update_classification_display()
    shot(w, "15_classified_no_samples")
    w.blend_toggle_btn.setChecked(True); w.toggle_blend_mode()
    w.opacity_slider.setValue(60)
    pump(0.5)
    shot(w, "16_blend_overlay")
    w.blend_toggle_btn.setChecked(False); w.toggle_blend_mode()

    # Accuracy tab with the workspace's 100 labelled points
    w.tabs.setCurrentWidget(w.accuracy_tab)
    w.acc_viewer.fit_view()
    shot(w, "17_accuracy_overview")
    w.acc_point_list.setCurrentRow(4)
    w.on_accuracy_point_list_double_clicked(w.acc_point_list.item(4))
    pump(0.5)
    shot(w, "18_accuracy_point_zoom")

    w.close()
    app.quit()


if __name__ == "__main__":
    main()
