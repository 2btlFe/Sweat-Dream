"""DINOv2 patch features of camera views and a fixed 3-component PCA basis over them.

Dream-1's API returns actions only, so these are not Dream-1's own features: they show how a general
self-supervised vision model (DINOv2 ViT-S/14) groups each view, patch by patch. The first three PCA components of
the patch features become red, green and blue; patches that look alike get alike colours.

pca.py fits one basis per episode and naf_upsampler.py lifts the 16 x 16 maps to 224 x 224.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

MODEL = "vit_small_patch14_dinov2.lvd142m"
PATCH = 14
INPUT_HW = (224, 224)  # 16 x 16 patches
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
VIEWS = ("top", "left", "right")


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class FeaturePCA:
    """DINOv2 patch features of a batch of views, projected on a fixed 3-component PCA basis."""

    def __init__(self, device: str | None = None, input_hw=INPUT_HW):
        import timm

        self.device = pick_device(device)
        self.model = timm.create_model(MODEL, pretrained=True, dynamic_img_size=True).eval().to(self.device)
        self.input_hw = tuple(input_hw)
        self.grid = (self.input_hw[0] // PATCH, self.input_hw[1] // PATCH)
        self.basis = None  # (mean [C], components [C, 3], lo [3], hi [3])

    @torch.inference_mode()
    def features(self, frames: list[np.ndarray]) -> torch.Tensor:
        """Patch features [views, rows, cols, C] (on the CPU) of uint8 HxWx3 frames."""
        x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div_(255)
        x = F.interpolate(x, size=self.input_hw, mode="bilinear", align_corners=False, antialias=True)
        x = ((x - MEAN) / STD).to(self.device)
        tokens = self.model.forward_features(x)[:, self.model.num_prefix_tokens :]
        return tokens.reshape(len(frames), *self.grid, -1).float().cpu()

    def fit(self, feats: torch.Tensor) -> None:
        """Fit the basis on features [N, rows, cols, C]; each component's sign is fixed so colours do not flip."""
        flat = feats.reshape(-1, feats.shape[-1])
        mean = flat.mean(0)
        _, _, v = torch.pca_lowrank(flat - mean, q=3, center=False)
        projected = (flat - mean) @ v
        sign = torch.where(projected.max(0).values.abs() >= projected.min(0).values.abs(), 1.0, -1.0)
        v, projected = v * sign, projected * sign
        lo, hi = torch.quantile(projected, 0.01, dim=0), torch.quantile(projected, 0.99, dim=0)
        self.basis = (mean, v, lo, hi)

    def project_float(self, feats: torch.Tensor) -> np.ndarray:
        """RGB maps [views, rows, cols, 3] as float32 in 0..1 (fits the basis first if there is none)."""
        if self.basis is None:
            self.fit(feats)
        mean, v, lo, hi = self.basis
        rgb = ((feats - mean) @ v - lo) / (hi - lo).clamp_min(1e-6)
        return rgb.clamp(0, 1).numpy().astype(np.float32)
