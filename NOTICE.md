# Lite Classify — copyright and licence notice

Lite Classify (internal name PyOBIA), an object-based image analysis application for drone
orthomosaics.

Copyright (C) 2025–2026 L. D. Barua (LeoBar). All rights not granted below are reserved.

This program is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation, version 3 of
the License (see `LICENSE`), together with the additional terms below.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

## Additional terms (GNU GPL version 3, section 7)

These terms apply to Lite Classify and to every work based on it.

1. **Author attribution (section 7(b)).** Any conveyed copy or modified version must keep
   the following attribution, unaltered, in the program's Appropriate Legal Notices (the
   About dialog or equivalent, and the startup console banner where one exists), and in the
   README or main documentation of the distribution:

   > Based on Lite Classify by L. D. Barua (LeoBar) — https://github.com/leobarua/liteclassify

2. **Marking of modified versions (section 7(c)).** A modified version must be marked as
   changed in reasonable ways, and must not be presented as the original Lite Classify or as
   endorsed by its author.

3. **Names and marks (section 7(e)).** This licence grants no right to use the names
   "Lite Classify", "LeoBar" or the Lite Classify icon to name or promote a modified version. A
   modified version must use a different name; it may describe itself as "based on Lite
   Classify" as required by term 1.

## Citing Lite Classify

If you publish maps, data, reports or papers produced with Lite Classify, please cite it as
given in `CITATION.cff` (GitHub: "Cite this repository"). This is a request, not a licence
condition: the GPL does not cover the output of the program.

## Other licensing

The author can licence Lite Classify under terms other than the GPL (for example, for use
inside a closed-source product). See `COMMERCIAL-LICENSING.md`.

## Third-party components

| Component | Where | Licence |
| --- | --- | --- |
| DINOv2 ViT backbone code (Meta Platforms, Inc.) | `third_party/dinov2/` | Apache License 2.0 (`third_party/dinov2/LICENSE`) |
| DINOv2 ViT-S/14 pretrained weights | bundled in builds as `models/dinov2_vits14_pretrain.pth` | Apache License 2.0 |

Lite Classify includes no photogrammetry software. Import > Orthomosaic from Drone Photos talks
to a WebODM installation (AGPL-3.0, OpenDroneMap) that the user installs and runs separately,
over WebODM's HTTP API; no WebODM, NodeODM or ODM code is part of Lite Classify.

Python dependencies (PySide6 — LGPL-3.0; NumPy, SciPy, scikit-image, scikit-learn,
PyTorch, rasterio, GDAL, psutil — BSD/MIT-style licences) are installed separately and keep
their own licences; binary builds include them under those licences.
