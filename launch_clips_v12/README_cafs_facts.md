# Launch deck v1.2: clips and CAFS facts

## Clips (this folder)

Each `_loop.mp4` replaces the video of the same name on its slide. The three new slides have grey placeholder frames that name their clip.

| Slide | Clip | What changed |
|---|---|---|
| s01 | C0_loop | CAFS ortho, zoom to one pixel |
| s02 | C1_loop | CAFS DSM terrain |
| s03 | C2_loop | CAFS objects (greenhouse) |
| s04 | C3a_loop | SLIC on a CAFS patch |
| s05 | C3b_loop | 30,000 / 67,536 / 150,000 segments |
| **new, after s05** | X1_loop | Process This Extent Only, with points kept inside the box |
| F1 slide | F1_feature_showcase | 44 tiles from the CAFS crop, DINOv3 ViT-S+/16 |
| s07 | C4a_loop | indices on CAFS |
| s08 | C4b_loop | GLCM: CAFS tree vs roof |
| s09 | C4c_loop | height above ground |
| s10 | C5a_loop | Crop 1 vs Crop 2 patches, 14 × 14 tiles |
| s11 | C5b_loop | DINOv3 embedding of 2,038 training objects |
| s12 | C6_loop | 36 discovered groups |
| s14 | C7a_loop | 9 classes, forest vote |
| **new, after s14** | S1_loop | cross-validated separability |
| s15 | C7b_loop | replay 83% → 90% → 92% |
| **new, after s15** | G1_loop | greenness refinement |
| s16 | C8_loop | the 450 real points, 9 × 9 matrix |
| s17 | C9_loop | 9 stations, CAFS |
| s_finale | F2_workflow_finale_loop | 44 bars, 9-class check |

## CAFS numbers for the notes

- Site: the CAFS experimental agricultural site, UPLB, Los Baños, Laguna.
- Area: 13.5 ha. Native pixel: 2.16 cm, about 290 million pixels over the mapped area. DSM: 8.6 cm.
- Segmentation: 5 cm working grid, 67,536 requested → 74,736 objects, about 1.8 m² each. Compactness 0.15, sigma 2.5.
- Colour and texture are read from the 2.16 cm pixels, about 5 times as many pixels as the 5 cm grid.
- Segment sizes over a patch: 30,000 requested ≈ 4.7 m², 67,536 ≈ 2 m², 150,000 ≈ 0.9 m².
- Features: 44 = 24 hand-crafted + 20 DINOv3 components (10 each from the 5 m and 10 m patches).
- GLCM contrast: tree 1.45 vs roof 0.41, about 3.6 times.
- DINOv3 ViT-S+/16: 196 tiles per patch (DINOv2 /14: 256), 12 layers, 384 numbers. User-supplied weights; DINOv2 ships with the app.
- Crop 1 vs Crop 2: the five most telling features are all appearance components. The two crops cover about 17% of the map.
- Discover: 30 requested → 36 groups, 20 named. Five groups became Bare; one group was split into Water and Mud.
- Training: 2,038 objects (2.7% of objects), 9 classes.
- Replay (10 random starts, 5 per class, then 8 more per class each round): round 1 83% (76–90%), round 3 90%, round 8 92%.
- Separability (cross-validated): Bare–Road 1.76, Bare–Grass 1.80, Crop 2–Grass 1.98, all others about 2.0.
- Greenness refinement: Grass/Bare, cutoff 0.042, 3 × 3 window, "Keep green segments whole". Accuracy 91.3% → 94.2%.
- Accuracy: 450 points, 50 per class, stratified. 93.6% plain, 94.2% ± 3.1% area-weighted. Points were moved short distances (median 1.7 m) for a clearer view. Training samples were refined while these points were in use, so the figure may be somewhat optimistic.
- No minimum mapping unit in the final map (a 10 m² sieve gives 93.1%).
