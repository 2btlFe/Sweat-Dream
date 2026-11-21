import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

NATTEN_AVAILABLE = True
try:
    NATTEN_RECENT = False
    from natten.functional import na2d_av, na2d_qk
except Exception:
    NATTEN_RECENT = True
    try:
        from natten import na2d
    except Exception:  # Clean-Data-Better-Dream: no NATTEN (macOS, MPS, CPU) -> the pure PyTorch version below
        NATTEN_AVAILABLE = False

from src.layers.rope import RoPE


def _na_indices(length, kernel, dilation, device):
    """NATTEN's dilated neighbourhoods along one axis: for each position, the `kernel` positions it attends to.

    Positions split into `dilation` groups by index modulo `dilation`; inside its group a position attends to the
    `kernel` group members centred on it, the window shifted (not padded) at the group's edges.
    """
    i = torch.arange(length, device=device)
    group, inner = i % dilation, i // dilation
    group_length = (length - group + dilation - 1) // dilation
    if int(group_length.min()) < kernel:
        raise ValueError(f"kernel {kernel} is larger than a dilation group ({int(group_length.min())})")
    start = torch.minimum(torch.clamp(inner - kernel // 2, min=0), group_length - kernel)
    return group[:, None] + dilation * (start[:, None] + torch.arange(kernel, device=device))


def na2d_torch(q, k, v, kernel_size, dilation, scale=None, rows_per_chunk=8):
    """Dilated 2D neighbourhood attention in plain PyTorch (works on CPU and MPS), NATTEN's `na2d` semantics.

    q, k: (B, H, W, heads, d); v: (B, H, W, heads, dv). Returns (B, H, W, heads, dv).
    """
    batch, height, width, heads, dim = q.shape
    kh, kw = kernel_size
    dh, dw = dilation
    scale = dim**-0.5 if scale is None else scale
    rows = _na_indices(height, kh, dh, q.device)
    cols = _na_indices(width, kw, dw, q.device)
    out = q.new_empty(batch, height, width, heads, v.shape[-1])
    for r0 in range(0, height, rows_per_chunk):
        r = rows[r0 : r0 + rows_per_chunk]  # (c, kh)
        keys = k[:, r][:, :, :, cols]  # (B, c, kh, W, kw, heads, d)
        values = v[:, r][:, :, :, cols]
        logits = torch.einsum("bcwnd,bcxwynd->bcwxyn", q[:, r0 : r0 + len(r)], keys) * scale
        weights = logits.flatten(3, 4).softmax(dim=3).view_as(logits)
        out[:, r0 : r0 + len(r)] = torch.einsum("bcwxyn,bcxwynd->bcwnd", weights, values)
    return out


def na2d_blocks(q, k_low, v_low, kernel_size, scale=None):
    """`na2d_torch` for NAF's case, computed per low-resolution cell: the same result, far faster.

    NAF's keys and values are a low-resolution grid enlarged by an integer factor (nearest), and the dilation is
    that factor. Every query in one cell then attends to the same kernel x kernel low-resolution cells, so each
    window is gathered once per cell and all the cell's queries go through one matrix product.

    q: (B, H, W, heads, d) with H, W multiples of k_low's grid; k_low: (B, h, w, heads, d); v_low: (B, h, w, heads, dv).
    """
    batch, height, width, heads, dim = q.shape
    _, low_h, low_w, _, dv = v_low.shape
    kh, kw = kernel_size
    fy, fx = height // low_h, width // low_w
    scale = dim**-0.5 if scale is None else scale
    rows = _na_indices(low_h, kh, 1, q.device)  # (h, kh)
    cols = _na_indices(low_w, kw, 1, q.device)  # (w, kw)

    def windows(t):  # (B, h, w, heads, c) -> (B, h, w, heads, kh*kw, c)
        t = t[:, rows][:, :, :, cols]  # (B, h, kh, w, kw, heads, c)
        return t.permute(0, 1, 3, 5, 2, 4, 6).reshape(batch, low_h, low_w, heads, kh * kw, t.shape[-1])

    cells = q.reshape(batch, low_h, fy, low_w, fx, heads, dim).permute(0, 1, 3, 5, 2, 4, 6)
    cells = cells.reshape(batch, low_h, low_w, heads, fy * fx, dim)
    weights = (cells @ windows(k_low).transpose(-1, -2) * scale).softmax(dim=-1)  # (B, h, w, heads, fy*fx, K)
    out = (weights @ windows(v_low)).reshape(batch, low_h, low_w, heads, fy, fx, dv)
    return out.permute(0, 1, 4, 2, 5, 3, 6).reshape(batch, height, width, heads, dv)


def na2d_blocks_spatial(q, k_low, v_low, heads, kernel_size, scale=None):
    """`na2d_blocks` on NAF's channel-first tensors, with one copy of q and of the output instead of two each.

    q: (B, heads*d, H, W); k_low: (B, heads*d, h, w); v_low: (B, heads*dv, h, w). Returns (B, heads*dv, H, W).
    Windows are gathered with index_select, much faster than advanced indexing on MPS.
    """
    batch, channels, height, width = q.shape
    dim = channels // heads
    low_h, low_w = k_low.shape[-2:]
    dv = v_low.shape[1] // heads
    kh, kw = kernel_size
    fy, fx = height // low_h, width // low_w
    scale = dim**-0.5 if scale is None else scale
    rows = _na_indices(low_h, kh, 1, q.device).flatten()
    cols = _na_indices(low_w, kw, 1, q.device).flatten()

    def windows(t, c):  # (B, heads*c, h, w) -> (B, h, w, heads, kh*kw, c)
        t = t.reshape(batch, heads, c, low_h, low_w).index_select(3, rows).index_select(4, cols)
        t = t.view(batch, heads, c, low_h, kh, low_w, kw).permute(0, 3, 5, 1, 4, 6, 2)
        return t.reshape(batch, low_h, low_w, heads, kh * kw, c)

    cells = q.view(batch, heads, dim, low_h, fy, low_w, fx).permute(0, 3, 5, 1, 4, 6, 2)
    cells = cells.reshape(batch, low_h, low_w, heads, fy * fx, dim)
    weights = (cells @ (windows(k_low, dim).transpose(-1, -2) * scale)).softmax(dim=-1)
    out = (weights @ windows(v_low, dv)).view(batch, low_h, low_w, heads, fy, fx, dv)
    return out.permute(0, 3, 6, 1, 4, 2, 5).reshape(batch, heads * dv, height, width)


def legacy_attention(q, k, v, kernel_size, dilation, scale=1, return_weights=False):
    q = rearrange(q, "b h w n d -> b n h w d")
    k = rearrange(k, "b h w n d -> b n h w d")
    v = rearrange(v, "b h w n d -> b n h w d")
    attn_scores = na2d_qk(q, k, kernel_size=kernel_size, dilation=dilation)
    attn_scores = attn_scores * scale

    attn_weights = attn_scores.softmax(dim=-1)
    features = na2d_av(attn_weights, v, kernel_size=kernel_size, dilation=dilation)
    features = rearrange(features, "b n h w d -> b h w n d")

    if return_weights:
        return features, attn_scores
    return features


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=(9, 9),
        **kwargs,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.kernel_size = kernel_size

        self.scale = (dim // num_heads) ** -0.5

    def _resize(self, x, size, dtype):
        x = F.interpolate(x, size=size, mode="nearest-exact")
        x = rearrange(x, "b (n d) h w -> b h w n d", n=self.num_heads)
        return x.to(dtype)

    def forward(self, q, k, v, image=None, return_weights=False, **kwargs):
        hq, wq = q.shape[-2:]
        hk, wk = k.shape[-2:]
        dilation = (hq // hk, wq // wk)
        self.dilation = dilation

        if (
            not NATTEN_AVAILABLE
            and not return_weights
            and hq % hk == 0
            and wq % wk == 0
            and hk >= self.kernel_size[0]
            and wk >= self.kernel_size[1]
        ):  # Clean-Data-Better-Dream: exact per-cell form of the neighbourhood attention below
            return na2d_blocks_spatial(
                q, k.to(q.dtype), v.to(q.dtype), self.num_heads, self.kernel_size, scale=self.scale
            )
        q = rearrange(q, "b (n d) h w -> b h w n d", n=self.num_heads)
        k = self._resize(k, size=(hq, wq), dtype=q.dtype)
        v = self._resize(v, size=(hq, wq), dtype=q.dtype)

        # Use legacy attention pattern
        if return_weights:
            assert not NATTEN_RECENT, "Return weights not supported with recent natten versions"
            out, attn_weights = legacy_attention(q, k, v, self.kernel_size, dilation, scale=self.scale, return_weights=True)
            return rearrange(out, "b h w n d -> b (n d) h w"), attn_weights
        else:
            if not NATTEN_AVAILABLE:
                out = na2d_torch(q, k, v, self.kernel_size, dilation, scale=self.scale)
            elif NATTEN_RECENT:
                # Use modern na2d attention
                # Note: Modern na2d doesn't support position bias directly
                out = na2d(q, k, v, kernel_size=self.kernel_size, dilation=dilation, stride=1, backend="cutlass-fna")
            else:
                out = legacy_attention(q, k, v, self.kernel_size, dilation, scale=self.scale)
            return rearrange(out, "b h w n d -> b (n d) h w")
