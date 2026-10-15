# Lite Classify

<img src="docs/img/brand/liteclassify_256.png" alt="Lite Classify icon" width="96" align="right">

**by LeoBar** · a lightweight desktop app for object-based land-cover mapping of drone orthomosaics.

Lite Classify picks up where [Lite Flight](https://leobarua.github.io/liteflight/) leaves off. Plan and fly the survey with Lite Flight, process the photos into an orthomosaic and surface model in your photogrammetry software, then bring both into Lite Classify to map them.

It splits the image into objects and describes each one by its colour, visible-light vegetation indices, texture, shape and height, plus a learned description of its appearance (DINOv2). It groups look-alike objects so you can discover and name the classes the scene contains, learns those classes from examples you click, classifies the whole scene with a Random Forest, and checks the finished map against an independent stratified sample with area-weighted accuracy estimates. It runs offline on one Windows workstation, with an optional NVIDIA GPU.

## Coming soon

Lite Classify will be released as free, open-source software under the GNU General Public License v3. The program and source code are not available for download yet. For early access, contact ldbarua@gmail.com.

## Manual

The illustrated user manual is at **https://leobarua.github.io/liteclassify/**. It walks one real 54.7-hectare drone survey from orthomosaic to accuracy report: getting started, how the method works (segmentation, the object features, DINOv2 appearance, class discovery, Random Forest, separability, retraining, accuracy assessment), every control tab by tab, and a reference section.

## About

Lite Classify is built by LeoBar for mapping canopy gaps, restoration patches and land use from drone surveys of tropical landscapes. The worked example in the manual is a mixed-use block in Bay, Laguna, Philippines, captured at 3.3 cm per pixel.
