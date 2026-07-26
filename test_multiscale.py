"""
Correctness check for the multiscale_layers architecture change in
MaskedEncoder/MaskedAutoencoderViT (src/smri_mae/model_mae.py).

Requires CUDA -- this codebase's packed/jagged-batch attention (nested
tensor SDPA) has no CPU backend, so this must be run on a GPU box.

Run: uv run python test_multiscale.py
"""

import sys
import torch

sys.path.insert(0, "src")
from smri_mae.model_mae import mae_vit_small

DEVICE = torch.device("cuda")
B, C = 2, 1
img_size = (32, 32, 32)  # small for a fast test
patch_size = 8


def make_input():
    torch.manual_seed(0)
    x = torch.randn(B, C, *img_size, device=DEVICE)
    mask = torch.ones(B, C, *img_size, dtype=torch.bool, device=DEVICE)
    mask[:, :, :8, :, :] = False  # simulate some missing/invalid voxels
    return x, mask


# --- Test 1: multiscale_layers=None must match baseline (within GPU kernel
# float tolerance -- see note below, this is not literal bit-exactness) ---
torch.manual_seed(1)
model_baseline = mae_vit_small(img_size=img_size, patch_size=patch_size, in_chans=C).to(DEVICE)
torch.manual_seed(1)
model_disabled = mae_vit_small(
    img_size=img_size, patch_size=patch_size, in_chans=C, multiscale_layers=None
).to(DEVICE)

x, mask = make_input()
torch.manual_seed(42)
loss_b, state_b = model_baseline(x, img_mask=mask, mask_ratio=0.75, with_state=True)
torch.manual_seed(42)
loss_d, state_d = model_disabled(x, img_mask=mask, mask_ratio=0.75, with_state=True)

# Note: torch.equal (bit-exact) can fail here even with identical weights and
# identical code paths, purely from GPU attention-kernel algorithm selection
# depending on allocator/context state between two separately-constructed
# model instances in the same process -- confirmed by showing two literally
# unmodified baseline models also disagree by the same tiny amount. Hence
# allclose with a loose-but-real tolerance, not exact equality.
assert torch.allclose(loss_b, loss_d, atol=1e-4, rtol=1e-4), (
    f"DISABLED-MODE MISMATCH: {loss_b} vs {loss_d}"
)
assert torch.allclose(state_b["pred_images"], state_d["pred_images"], atol=1e-4, rtol=1e-4), (
    "DISABLED-MODE pred mismatch"
)
print("[PASS] multiscale_layers=None matches baseline (within GPU float tolerance)")

# --- Test 2: multiscale_layers=[...] runs without error and produces correct shapes ---
torch.manual_seed(2)
model_ms = mae_vit_small(
    img_size=img_size,
    patch_size=patch_size,
    in_chans=C,
    multiscale_layers=[3, 7, 11],  # mae_vit_small has depth=12 -> valid indices 0..11
).to(DEVICE)
x, mask = make_input()
loss_ms, state_ms = model_ms(x, img_mask=mask, mask_ratio=0.75, with_state=True)
assert torch.isfinite(loss_ms), f"multiscale loss not finite: {loss_ms}"
assert state_ms["pred_images"].shape == state_b["pred_images"].shape, "shape mismatch vs baseline"
print(f"[PASS] multiscale_layers=[3,7,11] runs, loss={loss_ms.item():.4f}, shape OK")

# --- Test 3: gradient flows through the fusion path ---
model_ms.zero_grad()
loss_ms.backward()
fuse_grad_norm = model_ms.encoder.multiscale_fuse.weight.grad.norm().item()
assert fuse_grad_norm > 0, "no gradient reached multiscale_fuse -- fusion path is disconnected!"
print(f"[PASS] gradient reaches multiscale_fuse (grad norm={fuse_grad_norm:.4f})")

# --- Test 4: single-tap multiscale_layers=[11] (just the last block) runs ---
torch.manual_seed(2)
model_ms1 = mae_vit_small(
    img_size=img_size, patch_size=patch_size, in_chans=C, multiscale_layers=[11]
).to(DEVICE)
loss_ms1, _ = model_ms1(x, img_mask=mask, mask_ratio=0.75, with_state=True)
assert torch.isfinite(loss_ms1)
print(f"[PASS] single-tap multiscale_layers=[11] runs, loss={loss_ms1.item():.4f}")

print("\nALL TESTS PASSED")
