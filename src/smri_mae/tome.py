# This source code is licensed under the Apache License, Version 2.0

"""Token Merging (ToMe) for the MAE encoder.

Bolya et al., "Token Merging: Your ViT But Faster" (ICLR 2023),
https://arxiv.org/abs/2210.09461.

Per block, splits tokens into two interleaved sets, matches each token in one
set to its most similar token in the other (cosine similarity on the
attention keys, reused for free from the attention computation already run
that block), and merges the `r` most-similar pairs by a size-weighted
average. This only ever removes tokens (no new parameters), so pretrained
checkpoint weights load unmodified whether or not merging is enabled at
inference/training time.

Adapted for `smri_mae`'s packed jagged-batch encoder: instead of a fixed
`[B, N, D]` layout, all samples in a batch are packed into one `[L_total, D]`
tensor with `JaggedBatch` offsets marking per-sample boundaries (samples may
have different numbers of visible tokens to begin with, e.g. from sparse
volumes). Matching and merging must never mix tokens across samples, and
must never touch the leading `num_prefix_tokens` (cls/reg) tokens of each
sample's span. `tome_merge_packed` does this by slicing each sample out of
the packed tensor, merging its patch-token span independently (plain,
unbatched bipartite_soft_matching), and re-packing the result with fresh
offsets (per-sample token counts can differ, e.g. once a sample runs too low
on patch tokens to keep merging `r` pairs).
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from .modules import JaggedBatch

# (unm_idx, src_idx, dst_idx, r): index tensors describing one merge step, or
# None when no merge should happen (r <= 0 or too few tokens to merge).
Match = tuple[Tensor, Tensor, Tensor, int] | None


def bipartite_soft_matching(metric: Tensor, r: int) -> Match:
    """Rank token pairs to merge within a single sample's tokens.

    metric: [N, D] similarity features (e.g. mean attention keys).
    r: number of token pairs to merge (removes exactly r tokens), clamped to N // 2.
    """
    N = metric.shape[0]
    r = min(r, N // 2)
    if r <= 0:
        return None

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        a, b = metric[0::2], metric[1::2]
        scores = a @ b.transpose(-1, -2)

        node_max, node_idx = scores.max(dim=-1)
        edge_order = node_max.argsort(descending=True)

        unm_idx = edge_order[r:]
        src_idx = edge_order[:r]
        dst_idx = node_idx[src_idx]

    return unm_idx, src_idx, dst_idx, r


def apply_merge(x: Tensor, match: Match, reduce: str = "mean") -> Tensor:
    """Merge x: [N, ...] according to a bipartite_soft_matching result."""
    if match is None:
        return x
    unm_idx, src_idx, dst_idx, _ = match
    src, dst = x[0::2], x[1::2]
    d = src.shape[-1]
    unm = src[unm_idx]
    src_m = src[src_idx]
    dst = dst.scatter_reduce(
        0, dst_idx.unsqueeze(-1).expand(-1, d), src_m, reduce=reduce, include_self=True
    )
    return torch.cat([unm, dst], dim=0)


def merge_wavg(x: Tensor, size: Tensor, match: Match) -> tuple[Tensor, Tensor]:
    """Size-weighted merge: average by how many original tokens each entry represents."""
    if match is None:
        return x, size
    x = apply_merge(x * size, match, reduce="sum")
    size = apply_merge(size, match, reduce="sum")
    return x / size, size


def merge_pos_ids(pos_ids: Tensor, match: Match) -> Tensor:
    """Carry token position ids through a merge.

    Merged-away (src) tokens are simply dropped; a token that receives a merge
    (dst) keeps reporting its own original position — the decoder only uses
    this as approximate positional context for the token, not as a claim that
    it covers every position it absorbed.
    """
    if match is None:
        return pos_ids
    unm_idx, _, _, _ = match
    pos_src, pos_dst = pos_ids[0::2], pos_ids[1::2]
    return torch.cat([pos_src[unm_idx], pos_dst], dim=0)


def tome_merge_packed(
    x: Tensor,
    metric: Tensor,
    size: Tensor,
    pos_ids: Tensor,
    jagged_batch: JaggedBatch,
    num_prefix_tokens: int,
    r: int,
) -> tuple[Tensor, Tensor, Tensor, JaggedBatch]:
    """Merge r token pairs per sample in a packed [L_total, D] jagged batch.

    Only the patch-token span of each sample (i.e. everything after the
    leading `num_prefix_tokens` cls/reg tokens) is eligible for merging.
    Returns (new_x, new_size, new_pos_ids, new_jagged_batch); per-sample
    lengths in the new jagged batch may differ across samples even if they
    started equal, since a sample can run out of patch tokens to merge.
    """
    offsets = jagged_batch.offsets.tolist()
    new_x, new_size, new_pos = [], [], []
    new_counts = []

    for start, end in zip(offsets[:-1], offsets[1:]):
        row_x = x[start:end]
        row_metric = metric[start:end]
        row_size = size[start:end]
        row_pos = pos_ids[start:end]

        prefix_x, patch_x = row_x[:num_prefix_tokens], row_x[num_prefix_tokens:]
        patch_metric = row_metric[num_prefix_tokens:]
        prefix_size, patch_size = row_size[:num_prefix_tokens], row_size[num_prefix_tokens:]
        prefix_pos, patch_pos = row_pos[:num_prefix_tokens], row_pos[num_prefix_tokens:]

        match = bipartite_soft_matching(patch_metric, r)
        merged_patch_x, merged_patch_size = merge_wavg(patch_x, patch_size, match)
        merged_patch_pos = merge_pos_ids(patch_pos, match)

        new_x.append(torch.cat([prefix_x, merged_patch_x], dim=0))
        new_size.append(torch.cat([prefix_size, merged_patch_size], dim=0))
        new_pos.append(torch.cat([prefix_pos, merged_patch_pos], dim=0))
        new_counts.append(prefix_x.shape[0] + merged_patch_x.shape[0])

    out_x = torch.cat(new_x, dim=0)
    out_size = torch.cat(new_size, dim=0)
    out_pos = torch.cat(new_pos, dim=0)

    counts = torch.tensor(new_counts, device=x.device, dtype=torch.long)
    new_offsets = F.pad(counts.cumsum(dim=0), (1, 0))
    new_jagged_batch = JaggedBatch(offsets=new_offsets, max_seqlen=int(counts.max().item()))
    return out_x, out_size, out_pos, new_jagged_batch
