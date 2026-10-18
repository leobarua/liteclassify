# Manual figure tools

Everything in `docs/manual/img/` (except `img/brand/`) and every number in
`docs/manual/manual_data.js` comes from the Bay, Laguna survey's saved workspace
(`Phileco_ortho_workspace/`: 10 cm grid, 63,645 segments, 622 training objects, 7 classes,
514 stratified reference points), plus `Phileco_ortho.tif` and `Phileco_dem.tif`.
Point `PHILECO_DIR` at the folder holding those three (default
`D:/GXDX_Paper/02_Data/Aerial/Phileco_Leo`). Run from the repository root:

    python docs/manual/tools/build_figures_phileco.py          # everything (~5 min, a Qt window may flash)
    python docs/manual/tools/build_figures_phileco.py --data-only   # rewrite manual_data.js only

| Script | Does |
| --- | --- |
| `phileco_common.py` | Reads the workspace files directly (no engine run); caches the ortho and DSM on the 10 cm grid in `.cache/`. |
| `build_figures_phileco.py` | All data figures, the comparison-gallery crops, Discover group maps and tile strips, the separability table and 21 pair plots (rendered with the app's own `obia_ui.render_separability_image`), the simulated retraining replay, and `manual_data.json` / `manual_data.js`. |
| `shoot_app.py` | Drives the app window and saves screenshots. Written for the earlier campus sample; the starting point for re-shooting app screenshots. |
| `make_icon.py` | The Lite Classify icon. |

Notes

- The map in every figure is `classified_map.npy`, the map the workspace's
  `accuracy_report.csv` scored (all 514 point predictions match it). The exported
  `Phileco_classified.tif` (2026-10-14) predates the last retraining and differs from it.
- The script checks its own area-weighted estimator against the report
  (`OA check 0.896 0.896` at the end of the run).
- Refitting the Random Forest on the saved 622 samples reproduces the saved map on 97% of
  pixels (88.6% area-weighted OA vs the report's 89.6%): samples were edited after the last
  Classify. The replay charts use the refit as "your model"; the matrix uses the report.
- The retraining charts are a **simulated replay** built from the workspace's labels: round
  1 = 5 random objects per class; each round adds up to 8 per class that the current model
  misclassifies; 10 random starts; every round scored as area-weighted OA on the stratified
  points. Label it as simulated.
- The strata are the 9-class map at sampling time (Broadleaf trees, Coconut, Banana, Brush,
  …); the estimator handles strata that differ from the 7 final classes (Stehman 2014).
- App screenshots of the four-tab layout are not yet re-shot; the pages describe the window
  in text and tables until then.
- After regenerating, rebuild the manual site bundle (`releases/`) before publishing.
