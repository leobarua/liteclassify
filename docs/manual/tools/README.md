# Manual figure tools

Everything in `docs/manual/img/` (except `img/brand/` and `img/webodm_window.jpg`) and every
number in `docs/manual/manual_data.js` comes from the CAFS worked example: its saved workspace
(5 cm grid, 74,736 segments, 2,038 training objects, 9 classes) and its 450 stratified reference
points. Run from the repository root:

    python docs/manual/tools/build_figures_cafs.py              # everything (~5 min, a Qt window may flash)
    python docs/manual/tools/build_figures_cafs.py --data-only  # rewrite manual_data.js only

By default the script reads two local, untracked folders under
`experiments/manual_v12/stratified_draft/`:

- `CAFS_WS` (default `stratified_draft_v2/CAFS_Ortho_workspace`): the workspace. The ortho and DSM
  paths come from its `session.json`.
- `CAFS_RESULTS` (default `results_v2_final`): `final_reference_points.gpkg` and `final_map_v2.npy`.

| Script | Does |
| --- | --- |
| `cafs_common.py` | Reads the workspace and results; caches the ortho and DSM on the 5 cm grid in `.cache/`. |
| `build_figures_cafs.py` | All data figures, the comparison-gallery crops, Discover group maps and tile strips, the separability table and 36 pair plots (rendered with the app's own `obia_ui.render_separability_image`), the simulated retraining replay, and `manual_data.json` / `manual_data.js`. |
| `shoot_app.py` | Drives the app window and saves screenshots. Written for an earlier sample; the starting point for re-shooting app screenshots. |
| `make_icon.py` | The Lite Classify icon. |

Notes

- The map in every figure is the final refined map: the workspace's `classified_map.npy` (the
  object map) plus the greenness refinement (grass/bare, cutoff 0.042, 3 x 3, keep green segments
  whole), with no sieve. The script prints whether the app's own refinement reproduces it.
- The accuracy numbers come from the app's own `run_accuracy_assessment`. The script checks them
  with its own area-weighted estimator (`OA check` at the end of the run).
- The strata are the map at sampling time, seed 20261102, 50 per class. The samples were refined
  afterwards, and the estimates keep the original strata as weights (Stehman 2014). 229 of the 450
  points were moved a median of 1.7 m by the interpreter, without reference to the classified map.
- The retraining charts are a **simulated replay** built from the workspace's labels: round
  1 = 5 random objects per class; each round adds up to 8 per class that the current model
  misclassifies; 10 random starts; every round scored, with the greenness refinement, as
  area-weighted OA on the stratified points. Label it as simulated. A refit on all the samples
  reproduces the saved object map.
- After regenerating, rebuild the manual site bundle (`releases/`) before publishing.
