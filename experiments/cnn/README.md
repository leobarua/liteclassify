# CNN / DINOv2 experiments

Offline scripts (not used by the app) that tested whether pretrained image embeddings of each
segment's surroundings help the Random Forest. They led to the app's DINOv2 appearance
features and the Discover tab.

| Script | Question |
| --- | --- |
| `embed.py` | Shared code: cut a context patch around each segment and embed it with a frozen backbone |
| `exp1_campus.py` | Plumbing test: app features vs app features + embeddings, on the same reference points |
| `exp2_prepare.py`, `exp2_analyze.py`, `exp2_figures.py` | Without labels: do embeddings encode canopy pattern rather than illumination and terrain? |
| `exp3_tindalo.py` | Grass / brush / trees: spatially blocked cross-validation of several feature sets |
| `exp4_tindalo_refpoints.py` | The same feature sets scored on independent reference points |
| `exp5_clusters.py` | Cluster-then-label: how well over-clustered groups match labeled segments |
| `make_label_layers.py` | Labeling layers for QGIS |

The survey data and workspaces they read (`data/...`) are not part of the repository.
Outputs go to `out/` (ignored by git). Run from the repository root.
