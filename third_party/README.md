# Vendored third-party code

`dinov2/` - the ViT backbone and layers from Meta's DINOv2
(https://github.com/facebookresearch/dinov2), Apache License 2.0 (see `dinov2/LICENSE`).
Only `models/vision_transformer.py` and `layers/` are kept; `models/__init__.py` is reduced
to importing the backbone. Used by `obia_discover.py` for the Discover Classes tab.
The weights (`dinov2_vits14_pretrain.pth`, ~85 MB, Apache 2.0) are not in git: they are
bundled with the build in `models/`, or read from the torch hub cache, or downloaded once.

Nothing from DINOv3 is vendored here (Meta's DINOv3 License covers its code and weights). The
optional DINOv3 backbone uses timm's model code, installed as a dependency, and a weights file
the user supplies.
