"""PCA feature maps at full image resolution with NAF (Neighborhood Attention Filtering, CVPR 2026).

DINOv2 ViT-S/14 sees each view at 224 x 224 and gives a 16 x 16 grid of 384-d patch features. NAF upsamples them
zero-shot to 224 x 224, guided by the image: every pixel attends to the 9 x 9 patches around it, with weights
from the image's own content and position (RoPE), so feature edges follow the edges in the picture.

PCA is linear, and NAF's output for each of its 4 attention heads is a weighted average (softmax weights) of that
head's low-resolution values. So projecting first and upsampling the projection gives the same maps as upsampling
the 384-d features and projecting after: each head's 96 channels go through the PCA basis to 3 numbers, NAF
upsamples those 4 x 3 channels, the heads are summed and the mean's projection subtracted. 12 channels instead of
384 cost far less on a laptop.

The NAF code is vendored in third_party/naf (Apache-2.0, see its NOTICE); without NATTEN it runs a plain PyTorch
neighbourhood attention. Weights come from the NAF release on first use (torch hub cache).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
WEIGHTS = "https://github.com/valeoai/NAF/releases/download/model/naf_release.pth"
SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_naf(device):
    """The released NAF, as published (no speed-ups)."""
    if str(HERE / "third_party" / "naf") not in sys.path:
        sys.path.insert(0, str(HERE / "third_party" / "naf"))
    from src.model.naf import NAF

    model = NAF().to(device).eval()
    state = torch.hub.load_state_dict_from_url(WEIGHTS, progress=False, map_location=device)
    model.load_state_dict(state)
    return model


# ------------------------------------------------------------------ speed-ups
#
# Measured on an M3 (MPS), three 224 x 224 views at once: the released model takes ~500 ms, nearly all in its image
# encoder (3x3 convolutions at full resolution, then RoPE). The changes below keep the weights:
#   1. RoPE: sin/cos computed once per size and applied in the spatial layout, without two full-size copies. Exact.
#   2. Reflect-padded convolutions: the padding done once, then a convolution without padding (MPS pads
#      "reflect" slowly inside the convolution). Exact.
#   3. The image encoder in float16 (the attention stays in float32). Mean change 2e-4 of the features.
#   4. torch.compile of the image encoder, where it works (a few seconds once per process).


def _fast_rope(self, x, layout="spatial"):
    import math

    b, c, h, w = x.shape
    n = self.num_heads
    d = c // n
    key = (h, w, x.dtype, x.device)
    if getattr(self, "_fast_key", None) != key:
        coords = self.create_coordinate(H=h, W=w)
        angles = (2 * math.pi * coords[:, :, None] / self.periods[None, None, :]).flatten(1, 2).tile(2)  # (h*w, d)
        self._fast_cs = (
            torch.cos(angles).T.reshape(1, 1, d, h, w).to(x.dtype),
            torch.sin(angles).T.reshape(1, 1, d, h, w).to(x.dtype),
        )
        self._fast_key = key
    cos, sin = self._fast_cs
    half = d // 2
    xv = x.view(b, n, d, h, w)
    x1, x2 = xv[:, :, :half], xv[:, :, half:]
    # rope_apply: x * cos + rotate_half(x) * sin, rotate_half(x) = [-x2, x1]
    out = torch.cat([x1 * cos[:, :, :half] - x2 * sin[:, :, :half], x2 * cos[:, :, half:] + x1 * sin[:, :, half:]], dim=2)
    return out.view(b, c, h, w)


def _split_reflect_padding(module: torch.nn.Module) -> None:
    """Every reflect-padded convolution becomes ReflectionPad2d followed by the same convolution without padding."""
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Conv2d) and child.padding_mode == "reflect" and any(child.padding):
            ph, pw = child.padding
            child.padding, child.padding_mode = (0, 0), "zeros"
            child._reversed_padding_repeated_twice = [0, 0, 0, 0]
            setattr(module, name, torch.nn.Sequential(torch.nn.ReflectionPad2d((pw, pw, ph, ph)), child))
        else:
            _split_reflect_padding(child)


def fast_naf(device, half: bool | None = None, compile: bool | None = None):
    """The released NAF with the speed-ups above. half/compile default to on for MPS and CUDA, off on CPU."""
    import os
    import types

    model = load_naf(device)
    accelerated = torch.device(device).type in ("mps", "cuda")
    half = accelerated if half is None else half
    compile = (accelerated and os.environ.get("DREAM_SENSORS_NAF_COMPILE", "1") != "0") if compile is None else compile
    encoder = model.image_encoder
    encoder.rope.forward = types.MethodType(_fast_rope, encoder.rope)
    _split_reflect_padding(encoder)
    if half:
        encoder.half()
    run_encoder = encoder
    if compile:
        try:
            run_encoder = torch.compile(encoder, dynamic=False)
        except Exception:
            run_encoder = encoder
    dtype = torch.float16 if half else torch.float32

    def forward(self, image, features, output_size, *args, **kwargs):
        x = run_encoder(image.to(dtype), output_size=output_size).float()
        keys = self.key_encoder(x, features)
        return self.upsampler(self.query_encoder(x), keys, features, image)

    model.forward = types.MethodType(forward, model)
    model.speedups = {"half": half, "compile": compile is not False and run_encoder is not encoder}
    return model


class NAFPCA:
    """224 x 224 PCA maps of square views: DINOv2 at 224 x 224, PCA, NAF upsampling of the projection."""

    def __init__(self, pca, device=None, fast: bool = True):
        """@param pca A feature_pca.FeaturePCA made with input_hw=(224, 224); its basis is used as fitted.
        @param fast The speed-ups below (fast_naf); False runs NAF as released."""
        self.pca = pca
        self.device = pca.device if device is None else torch.device(device)
        self.naf = fast_naf(self.device) if fast else load_naf(self.device)
        self.heads = self.naf.upsampler.num_heads

    @torch.inference_mode()
    def maps(self, frames: list[np.ndarray], feats: torch.Tensor | None = None) -> np.ndarray:
        """(views, 224, 224, 3) float RGB in 0..1 for uint8 frames; `feats` are their 16 x 16 features if known."""
        if feats is None:
            feats = self.pca.features(frames)  # (V, 16, 16, C) on the CPU
        mean, basis, lo, hi = self.pca.basis  # (C,), (C, 3), (3,), (3,)
        views, rows, cols, channels = feats.shape
        per_head = channels // self.heads
        # Each head's channels through its rows of the basis: (V, heads*3, 16, 16), laid out as NAF's (n d).
        f = feats.reshape(views, rows, cols, self.heads, per_head)
        b = basis.reshape(self.heads, per_head, 3)
        projected = torch.einsum("vrchd,hdk->vhkrc", f, b).reshape(views, self.heads * 3, rows, cols)
        image = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div_(255)
        image = F.interpolate(image, size=(SIZE, SIZE), mode="bicubic", align_corners=False).clamp_(0, 1)
        image = ((image - MEAN) / STD).to(self.device)
        up = self.naf(image, projected.to(self.device), (SIZE, SIZE)).float().cpu()  # (V, heads*3, 224, 224)
        rgb = up.reshape(views, self.heads, 3, SIZE, SIZE).sum(1) - (mean @ basis).view(1, 3, 1, 1)
        rgb = (rgb - lo.view(1, 3, 1, 1)) / (hi - lo).clamp_min(1e-6).view(1, 3, 1, 1)
        return rgb.clamp(0, 1).permute(0, 2, 3, 1).numpy().astype(np.float32)


def square(frame: np.ndarray) -> np.ndarray:
    """A view as the models see it here: the whole frame resized to 224 x 224 (squeezed, not cropped)."""
    from PIL import Image

    return np.asarray(Image.fromarray(frame).resize((SIZE, SIZE), Image.Resampling.BICUBIC))
