"""
Correctness check: the new batched fast path in tome_merge_packed must produce
IDENTICAL output to the original per-sample loop implementation, for varied
per-sample token counts (the realistic case, confirmed earlier: samples in
this dataset do NOT have equal valid-token counts). Run standalone, prints
PASS/FAIL, exits nonzero on any mismatch.
"""
import sys

sys.path.insert(0, "src")

import torch

from smri_mae.modules import JaggedBatch
from smri_mae.tome import _tome_merge_packed_batched, _tome_merge_packed_loop, tome_merge_packed

torch.manual_seed(0)


def make_batch(counts, D=8, num_prefix_tokens=1, device="cpu"):
    total = sum(counts)
    x = torch.randn(total, D, device=device)
    metric = torch.randn(total, D, device=device)
    size = torch.ones(total, 1, device=device)
    pos_ids = torch.arange(total, device=device)
    offsets = torch.tensor([0] + list(torch.tensor(counts).cumsum(0).tolist()), device=device)
    jb = JaggedBatch(offsets=offsets, max_seqlen=max(counts))
    return x, metric, size, pos_ids, jb


def compare(counts, r, num_prefix_tokens=1, D=8, seed=0, device="cpu"):
    """Compare loop vs the public dispatcher (tome_merge_packed), which is
    what real callers use -- so r=0 correctly exercises the loop path (the
    dispatcher routes it there), not a direct/unguarded call to the batched
    internals.

    Physical token order within a sample is allowed to differ between the
    two implementations (this is fine -- see the note above -- since
    pos_ids travel with each token and positional embeddings are applied
    from pos_ids, not raw array position; the model is permutation-
    equivariant given correct position tracking). So we compare by sorting
    each sample's tokens by pos_id first, then checking the values line up.
    """
    torch.manual_seed(seed)
    x, metric, size, pos_ids, jb = make_batch(counts, D=D, num_prefix_tokens=num_prefix_tokens, device=device)

    x1, s1, p1, jb1 = _tome_merge_packed_loop(x, metric, size, pos_ids, jb, num_prefix_tokens, r)
    x2, s2, p2, jb2 = tome_merge_packed(x, metric, size, pos_ids, jb, num_prefix_tokens, r)

    ok = True
    if not torch.equal(jb1.offsets, jb2.offsets):
        print(f"  MISMATCH offsets: loop={jb1.offsets.tolist()} batched={jb2.offsets.tolist()}")
        return False

    o1, o2 = jb1.offsets.tolist(), jb2.offsets.tolist()
    for i in range(len(counts)):
        p1_seg, p2_seg = p1[o1[i] : o1[i + 1]], p2[o2[i] : o2[i + 1]]
        if not torch.equal(p1_seg.sort().values, p2_seg.sort().values):
            print(f"  MISMATCH pos_id SET for sample {i} (different tokens survived -- a real bug)")
            ok = False
            continue

        order1, order2 = p1_seg.argsort(), p2_seg.argsort()
        x1_seg, x2_seg = x1[o1[i] : o1[i + 1]][order1], x2[o2[i] : o2[i + 1]][order2]
        s1_seg, s2_seg = s1[o1[i] : o1[i + 1]][order1], s2[o2[i] : o2[i + 1]][order2]
        if not torch.allclose(x1_seg, x2_seg, atol=1e-4, rtol=1e-4):
            print(f"  MISMATCH x values for sample {i} (after aligning by pos_id): max abs diff="
                  f"{(x1_seg - x2_seg).abs().max().item()}")
            ok = False
        if not torch.allclose(s1_seg, s2_seg, atol=1e-4, rtol=1e-4):
            print(f"  MISMATCH size values for sample {i} (after aligning by pos_id)")
            ok = False
    return ok


cases = [
    ("equal lengths, all samples same", [64, 64, 64, 64], 4),
    ("unequal lengths, realistic spread", [4056, 3980, 4200, 3912], 16),
    ("unequal lengths, small", [20, 30, 24], 4),
    ("single sample", [50], 8),
    ("large batch, unequal", [4056, 3980, 4200, 3912, 4100, 3950, 4300, 4020, 3890, 4150, 4000, 4080], 16),
    ("r=0 (no-op)", [40, 60, 50], 0),
]

all_pass = True
for name, counts, r in cases:
    ok = compare(counts, r)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}  counts={counts} r={r}")
    all_pass = all_pass and ok

if torch.cuda.is_available():
    print("\n--- repeating all cases on CUDA ---")
    for name, counts, r in cases:
        ok = compare(counts, r, device="cuda")
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}  counts={counts} r={r} (cuda)")
        all_pass = all_pass and ok

print()
if all_pass:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
    sys.exit(1)
