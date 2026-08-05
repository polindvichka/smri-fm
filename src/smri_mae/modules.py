# This source code is licensed under the Apache License, Version 2.0
#
# References:
# capi: https://github.com/facebookresearch/capi/blob/main/model.py
# timm: https://github.com/huggingface/pytorch-image-models/blob/v1.0.20/timm/models/vision_transformer.py
# vjepa2: https://github.com/facebookresearch/vjepa2/blob/main/src/models/utils/pos_embs.py

import math
from functools import partial
from typing import NamedTuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, reduce, repeat
from jaxtyping import Float, Int
from timm.layers import DropPath, to_3tuple

Layer = Type[nn.Module]


class JaggedBatch(NamedTuple):
    """Sequence boundaries and cached launch metadata for jagged attention."""

    offsets: Tensor
    max_seqlen: int

    @classmethod
    def from_mask(cls, mask: Tensor) -> "JaggedBatch":
        mask = mask.to(dtype=torch.bool)
        counts = mask.sum(dim=1)
        return cls(
            offsets=F.pad(counts.cumsum(dim=0), (1, 0)),
            max_seqlen=mask.shape[1],
        )

    def as_nested(self, tokens: Tensor) -> Tensor:
        # Cached conservative bounds avoid min/max reductions and GPU-to-CPU
        # synchronization when Flash SDPA inspects the jagged sequence lengths.
        return torch.nested.nested_tensor_from_jagged(
            tokens,
            self.offsets,
            min_seqlen=1,
            max_seqlen=self.max_seqlen,
        ).transpose(1, 2)


def unpack_tokens(tokens: Tensor, token_mask: Tensor) -> Tensor:
    """Restore packed values to a padded batch, filling invalid slots with zero."""
    output = tokens.new_zeros((*token_mask.shape, *tokens.shape[1:]))
    return output.index_put((token_mask,), tokens)


def _rotate_half_axial(x: Tensor, chunk_dim: int, num_axes: int = 3) -> Tensor:
    """rotate_half (standard RoPE (-x2, x1) swap) applied independently within
    each of `num_axes` chunks of size `chunk_dim` at the front of the last
    dim. Any trailing dims beyond num_axes*chunk_dim are left as zero here --
    apply_rope_3d's cos/sin padding makes those an identity op (x*1 + 0*0).

    x: [L, h, dh]
    """
    rotated_dim = chunk_dim * num_axes
    x_rot, x_rest = x[..., :rotated_dim], x[..., rotated_dim:]
    length, heads = x_rot.shape[0], x_rot.shape[1]
    x_rot = x_rot.view(length, heads, num_axes, chunk_dim)
    x1, x2 = x_rot.chunk(2, dim=-1)
    x_rot = torch.cat([-x2, x1], dim=-1).reshape(length, heads, rotated_dim)
    return torch.cat([x_rot, torch.zeros_like(x_rest)], dim=-1)


def _rope_cos_sin(
    positions: Float[Tensor, "L 3"],
    inv_freq: Tensor,
    chunk_dim: int,
    head_dim: int,
) -> tuple[Tensor, Tensor]:
    """positions: per-token 3D grid coordinate, [L, 3]. inv_freq: [chunk_dim // 2],
    shared frequency basis reused across all 3 spatial axes. Returns cos, sin
    each [L, 1, head_dim] -- axis chunks laid out consecutively
    [axis0 | axis1 | axis2 | identity remainder], matching _rotate_half_axial's
    layout, ready to broadcast against a [L, h, head_dim] q/k tensor."""
    length = positions.shape[0]
    half = chunk_dim // 2
    # [L, 3, half]
    angles = positions.unsqueeze(-1).float() * inv_freq.view(1, 1, half)
    # duplicate each axis's half-frequencies to fill its full chunk: [L, 3, chunk_dim]
    angles = torch.cat([angles, angles], dim=-1)
    angles = angles.reshape(length, 3 * chunk_dim)
    remainder = head_dim - 3 * chunk_dim
    if remainder > 0:
        angles = torch.cat([angles, angles.new_zeros(length, remainder)], dim=-1)
    cos = angles.cos().unsqueeze(1)
    sin = angles.sin().unsqueeze(1)
    return cos, sin


def apply_rope_3d(
    q: Tensor,
    k: Tensor,
    positions: Float[Tensor, "L 3"],
    inv_freq: Tensor,
    chunk_dim: int,
) -> tuple[Tensor, Tensor]:
    """3D-axial rotary position embedding: splits head_dim into 3 per-axis
    chunks (one per spatial dimension) and rotates each chunk of q/k by an
    angle proportional to that token's coordinate along the matching axis.
    Any head_dim remainder beyond 3*chunk_dim is left unrotated. q, k: [L, h, dh].
    """
    head_dim = q.shape[-1]
    cos, sin = _rope_cos_sin(positions, inv_freq, chunk_dim, head_dim)
    cos, sin = cos.to(q.dtype), sin.to(q.dtype)
    q_out = q * cos + _rotate_half_axial(q, chunk_dim) * sin
    k_out = k * cos + _rotate_half_axial(k, chunk_dim) * sin
    return q_out, k_out


def jagged_scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    jagged_batch: JaggedBatch,
) -> Tensor:
    """Run SDPA on a packed batch of variable-length sequences."""
    output_jagged = F.scaled_dot_product_attention(
        jagged_batch.as_nested(query),
        jagged_batch.as_nested(key),
        jagged_batch.as_nested(value),
    )
    return output_jagged.transpose(1, 2).values()


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        # 3D-axial RoPE: head_dim is split into 3 per-axis chunks (largest
        # even size <= head_dim // 3); any remainder dims are left unrotated.
        # See apply_rope_3d in this module for the actual rotation.
        self.use_rope = use_rope
        self.rope_chunk_dim = 2 * ((self.head_dim // 3) // 2) if use_rope else 0
        if self.rope_chunk_dim > 0:
            inv_freq = 1.0 / (
                rope_theta
                ** (torch.arange(0, self.rope_chunk_dim, 2).float() / self.rope_chunk_dim)
            )
            self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        else:
            self.rope_inv_freq = None

    def extra_repr(self):
        extra = f"num_heads={self.num_heads}"
        if self.use_rope:
            extra += f", rope_chunk_dim={self.rope_chunk_dim}"
        return extra

    def forward(
        self,
        x: Float[Tensor, "L D"],
        jagged_batch: JaggedBatch,
        positions: Float[Tensor, "L 3"] | None = None,
    ) -> Float[Tensor, "L D"]:
        L, D = x.shape
        h, dh = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(L, 3, h, dh)
        q, k, v = qkv.unbind(1)

        if self.rope_chunk_dim > 0 and positions is not None:
            q, k = apply_rope_3d(q, k, positions, self.rope_inv_freq, self.rope_chunk_dim)

        x = jagged_scaled_dot_product_attention(
            q,
            k,
            v,
            jagged_batch=jagged_batch,
        )
        x = x.reshape(L, D)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: int | float = 4,
        bias: bool = False,
    ) -> None:
        super().__init__()
        hidden_features = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, dim, bias=bias)

    def forward(self, x: Float[Tensor, "... D"]) -> Float[Tensor, "... D"]:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# timm default eps=1e-6
LayerNorm = partial(nn.LayerNorm, eps=1e-6)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        mlp_ratio: int | float = 4,
        drop_path: float = 0.0,
        norm_layer: Layer = LayerNorm,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            use_rope=use_rope,
            rope_theta=rope_theta,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            dim=dim,
            mlp_ratio=mlp_ratio,
            bias=proj_bias,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(
        self,
        x: Float[Tensor, "L D"],
        jagged_batch: JaggedBatch,
        positions: Float[Tensor, "L 3"] | None = None,
    ) -> Float[Tensor, "L D"]:
        x = x + self.drop_path1(
            self.attn(
                self.norm1(x),
                jagged_batch=jagged_batch,
                positions=positions,
            )
        )
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


# Patching and position embedding modules


class Patchify3D(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int, int],
        patch_size: int | tuple[int, int, int],
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        self.img_size = to_3tuple(img_size)
        self.patch_size = to_3tuple(patch_size)
        self.in_chans = in_chans

        T, H, W = self.img_size
        p_t, p_h, p_w = self.patch_size
        if T % p_t or H % p_h or W % p_w:
            raise ValueError(
                f"img_size {self.img_size} must be divisible by patch_size {self.patch_size}"
            )
        self.grid_size = (T // p_t, H // p_h, W // p_w)
        self.num_patches = math.prod(self.grid_size)
        self.patch_dim = in_chans * math.prod(self.patch_size)

    def forward(self, x: Float[Tensor, "B C T H W"]) -> Float[Tensor, "B N P"]:
        x = patchify3d(x, self.patch_size)
        return x

    def unpatchify(self, x: Float[Tensor, "B N P"]) -> Float[Tensor, "B C T H W"]:
        x = unpatchify3d(x, patch_size=self.patch_size, img_size=self.img_size)
        return x

    def extra_repr(self):
        return f"{self.img_size}, {self.patch_size}, in_chans={self.in_chans}"


def patchify3d(x: Tensor, patch_size: tuple[int, int, int]) -> Tensor:
    p_t, p_h, p_w = to_3tuple(patch_size)
    B, C, T, H, W = x.shape
    x = rearrange(x, "b c (t u) (h p) (w q) -> b (t h w) (c u p q)", u=p_t, p=p_h, q=p_w)
    return x


def unpatchify3d(
    x: Tensor,
    patch_size: tuple[int, int, int],
    img_size: tuple[int, int, int],
) -> Tensor:
    B, N, P = x.shape
    p_t, p_h, p_w = to_3tuple(patch_size)
    T, H, W = to_3tuple(img_size)
    x = rearrange(
        x,
        "b (t h w) (c u p q) -> b c (t u) (h p) (w q)",
        t=T // p_t,
        h=H // p_h,
        w=W // p_w,
        u=p_t,
        p=p_h,
        q=p_w,
    )
    return x


def avgpool_patch3d(
    x: Tensor,
    patch_size: tuple[int, int, int],
    in_chans: int,
    factor: int,
) -> Tensor:
    """Block-average a flattened patch (last dim, layout (c u p q)) down by
    `factor` per spatial axis. Any number of leading batch dims. Used to
    build the coarse ("big shape") reconstruction target from the same
    fine-resolution target the detail head is scored against."""
    u, p, q = patch_size
    x = rearrange(x, "... (c u p q) -> ... c u p q", c=in_chans, u=u, p=p, q=q)
    x = reduce(
        x,
        "... c (u f1) (p f2) (q f3) -> ... c u p q",
        "mean",
        f1=factor,
        f2=factor,
        f3=factor,
    )
    x = rearrange(x, "... c u p q -> ... (c u p q)")
    return x


def upsample_patch3d(
    x: Tensor,
    coarse_shape: tuple[int, int, int],
    in_chans: int,
    factor: int,
) -> Tensor:
    """Inverse of avgpool_patch3d: nearest-neighbor upsample a coarse,
    flattened patch prediction back to full patch resolution."""
    u, p, q = coarse_shape
    x = rearrange(x, "... (c u p q) -> ... c u p q", c=in_chans, u=u, p=p, q=q)
    x = repeat(
        x,
        "... c u p q -> ... c (u f1) (p f2) (q f3)",
        f1=factor,
        f2=factor,
        f3=factor,
    )
    x = rearrange(x, "... c u p q -> ... (c u p q)")
    return x


class CoarseToFineHead(nn.Module):
    """
    Predicts each masked patch in two stages, mirroring how a human sketches
    the overall shape before adding detail: `coarse_head` first guesses a
    block-averaged (downsampled by `coarse_factor` per axis) version of the
    patch, then `detail_head` predicts a fine residual on top of the
    (upsampled) coarse guess. The two are summed into the final prediction.

    Drop-in replacement for the plain nn.Linear decoder head used by the
    original model -- same call signature, Tensor in, Tensor out at full
    patch resolution -- so no other decoder/loss plumbing needs to know
    about it. The coarse guess itself is cached on `last_coarse_pred` after
    each forward() call so the training loop can also score it directly
    against a coarse-pooled target (the "sketch the big shape first" part of
    the loss, not just an emergent side effect of the residual sum).
    """

    def __init__(
        self,
        in_dim: int,
        patch_size: tuple[int, int, int],
        in_chans: int,
        coarse_factor: int = 2,
    ) -> None:
        super().__init__()
        if any(p % coarse_factor != 0 for p in patch_size):
            raise ValueError(
                f"patch_size {patch_size} must be divisible by coarse_factor {coarse_factor}"
            )
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.coarse_factor = coarse_factor
        self.coarse_shape = tuple(p // coarse_factor for p in patch_size)

        patch_dim = in_chans * math.prod(patch_size)
        coarse_dim = in_chans * math.prod(self.coarse_shape)

        self.coarse_head = nn.Linear(in_dim, coarse_dim)
        self.detail_head = nn.Linear(in_dim, patch_dim)
        self.last_coarse_pred: Tensor | None = None

    def extra_repr(self):
        return f"patch_size={self.patch_size}, coarse_factor={self.coarse_factor}"

    def forward(self, x: Tensor) -> Tensor:
        coarse = self.coarse_head(x)
        self.last_coarse_pred = coarse
        upsampled = upsample_patch3d(coarse, self.coarse_shape, self.in_chans, self.coarse_factor)
        return upsampled + self.detail_head(x)


class AbsolutePosEmbed(nn.Module):
    def __init__(self, embed_dim: int, grid_size: tuple[int, ...]) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.num_patches = math.prod(grid_size)

        self.weight = nn.Parameter(torch.empty(self.num_patches, embed_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(
        self,
        x: Float[Tensor, "B L D"],
        pos_ids: Int[Tensor, "B L"] | None = None,
    ) -> Float[Tensor, "B L D"]:
        x = apply_pos_embed(x, self.weight, pos_ids=pos_ids)
        return x

    def extra_repr(self):
        return f"{self.embed_dim}, {self.grid_size}"


class SeparablePosEmbed(nn.Module):
    def __init__(self, embed_dim: int, grid_size: tuple[int, ...]) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.num_patches = math.prod(grid_size)

        N_t, *grid_size_spatial = grid_size
        N_s = math.prod(grid_size_spatial)
        self.weight_spatial = nn.Parameter(torch.empty(1, N_s, embed_dim))
        self.weight_temporal = nn.Parameter(torch.empty(N_t, 1, embed_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight_spatial, std=0.02)
        nn.init.trunc_normal_(self.weight_temporal, std=0.02)

    def forward(
        self,
        x: Float[Tensor, "B L D"],
        pos_ids: Int[Tensor, "B L"] | None = None,
    ) -> Float[Tensor, "B L D"]:
        B, N, D = x.shape
        weight = (self.weight_temporal + self.weight_spatial).flatten(0, 1)  # [N, D]
        x = apply_pos_embed(x, weight, pos_ids=pos_ids)
        return x

    def extra_repr(self):
        return f"{self.embed_dim}, {self.grid_size}"


class SinCosPosEmbed3D(nn.Module):
    def __init__(self, embed_dim: int, grid_size: tuple[int, int, int]) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.num_patches = math.prod(grid_size)

        N_t, N_h, N_w = grid_size
        weight = get_3d_sincos_pos_embed(
            embed_dim=embed_dim,
            grid_size=(N_h, N_w),
            grid_depth=N_t,
            uniform_power=True,
        )
        self.weight = nn.Parameter(torch.from_numpy(weight).float(), requires_grad=False)

    def forward(
        self,
        x: Float[Tensor, "B L D"],
        pos_ids: Int[Tensor, "B L"] | None = None,
    ) -> Float[Tensor, "B L D"]:
        x = apply_pos_embed(x, self.weight, pos_ids=pos_ids)
        return x

    def extra_repr(self):
        return f"{self.embed_dim}, {self.grid_size}"


# sincos pos embed utils from vjepa2, but fixed the confusing meshgrid indexing


def get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=False):
    """
    grid_size: tuple of int of the grid height and width
    grid_depth: int of the grid depth
    returns:
        pos_embed: [grid_depth*grid_height*grid_width, embed_dim] (w/o cls_token)
                or [1+grid_depth*grid_height*grid_width, embed_dim] (w/ cls_token)
    """
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size[0], dtype=float)
    grid_w = np.arange(grid_size[1], dtype=float)
    grid_d, grid_h, grid_w = np.meshgrid(grid_d, grid_h, grid_w, indexing="ij")

    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim / 6) * 2)

    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)  # (T*H*W, D1)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)  # (T*H*W, D2)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)  # (T*H*W, D3)
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    pos_embed = pos_embed[:, :embed_dim]
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    returns: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def apply_pos_embed(
    x: Float[Tensor, "B L D"],
    weight: Float[Tensor, "N D"],
    pos_ids: Int[Tensor, "B L"] | None = None,
) -> Float[Tensor, "B L D"]:
    B, L, D = x.shape
    weight = weight.expand(B, -1, -1)
    if pos_ids is not None:
        weight = weight.gather(1, pos_ids.unsqueeze(-1).expand(-1, -1, D))
    x = x + weight
    return x


# (masked) normalization used for MAE target normalization


class Normalize(nn.Module):
    def __init__(
        self,
        grid_size: tuple[int, ...],
        dim: int | tuple[int, ...] | None = -1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.dim = dim
        self.eps = eps

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """
        Normalize input sequence along dim(s) after reshaping to grid.
        Returns tuple of (x, mean, std).
        """
        B, N, D = x.shape
        x = x.reshape((B, *self.grid_size, D))
        if mask is not None:
            mask = mask.reshape((B, *self.grid_size, D))
            x, mean, std = masked_normalize(x, mask, dim=self.dim, eps=self.eps)
        else:
            x, mean, std = normalize(x, dim=self.dim, eps=self.eps)
        mean = mean.expand_as(x).reshape(B, N, D)
        std = std.expand_as(x).reshape(B, N, D)
        x = x.reshape(B, N, D)
        return x, mean, std

    def extra_repr(self):
        return f"{self.grid_size}, dim={self.dim}"


def masked_normalize(
    x: Tensor,
    mask: Tensor,
    dim: int | tuple[int, ...] | None = -1,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    num_obs = mask.sum(dim=dim, keepdim=True).clamp(min=1)
    mean = (mask * x).sum(dim=dim, keepdim=True) / num_obs
    var = (mask * (x - mean) ** 2).sum(dim=dim, keepdim=True) / num_obs
    std = (var + eps) ** 0.5
    x = mask * (x - mean) / std
    return x, mean, std


def normalize(
    x: Tensor,
    dim: int | tuple[int, ...] | None = -1,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    mean = x.mean(dim=dim, keepdim=True)
    var = torch.var(x, dim=dim, keepdim=True, unbiased=False)
    std = (var + eps) ** 0.5
    x = (x - mean) / std
    return x, mean, std
