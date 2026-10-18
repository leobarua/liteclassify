"""CNN context-patch embeddings per segment (offline prototype, not used by the app).

For each segment, a square patch of `patch_m` metres centred on the segment centroid is cut
from the RGB ortho, resized to 224 px and passed through a frozen pretrained backbone. The
pooled feature vector is the segment's embedding."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_backbone(name):
    if name == "resnet50":
        import torchvision
        m = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        m.fc = torch.nn.Identity()
    elif name == "dinov2_s":
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    else:
        raise ValueError(name)
    return m.eval().cuda()


def centroids(seg):
    ids = np.unique(seg)
    ids = ids[ids > 0]
    c = ndimage.center_of_mass(np.ones_like(seg, dtype=np.uint8), seg, ids)
    return ids, np.array(c)  # (row, col)


@torch.no_grad()
def embed(ortho_chw, ids, cents, patch_px, model, batch=128):
    """ortho_chw: uint8 (3,H,W). Returns (len(ids), D) float32."""
    half = patch_px // 2
    padded = np.pad(ortho_chw, ((0, 0), (half, half), (half, half)), mode="reflect")
    out = []
    for i in range(0, len(ids), batch):
        crops = []
        for r, c in cents[i:i + batch]:
            r, c = int(round(r)), int(round(c))
            crops.append(padded[:, r:r + 2 * half, c:c + 2 * half])
        x = torch.from_numpy(np.stack(crops)).cuda().float() / 255.0
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        x = (x - MEAN.cuda()) / STD.cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            f = model(x)
        out.append(f.float().cpu().numpy())
    return np.concatenate(out)
