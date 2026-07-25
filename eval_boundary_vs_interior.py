"""
Fair three-way comparison of baseline vs coarse2fine vs edgeloss, split into
boundary voxels (high local intensity-gradient, i.e. tissue edges) vs interior
voxels (flat, easy-to-interpolate regions) -- this is the split edgeloss's own
training loss is designed around, so it's the fair way to judge whether the
edge-weighting trick actually achieved what it set out to do (plain whole-
volume MSE, computed separately, already showed edgeloss "losing" -- but that
metric treats boundary and interior voxels as equally important, which isn't
what edgeloss was optimizing for).

Method:
- Same val batches, same img_mask, for all three models (identical data).
- Same eval_seed reset before each model's forward pass, so all three see the
  identical random mask pattern per batch (same masking.py/patchify code
  across all three repos, confirmed via diff earlier).
- Boundary mask = top 20% of *predicted* (masked-out), *valid* voxels by
  local intensity-gradient magnitude, computed on the ground-truth image with
  the same gradient_magnitude_3d formula edgeloss's own loss uses.
- Report plain per-voxel MSE separately for boundary vs interior voxels, for
  each of the three checkpoints.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, "/workspace/smri-fm/src")
sys.path.insert(0, "/workspace/smri-fm-coarse2fine/src")

import data.mri_data as mri_data
import smri_mae.model_mae as baseline_models
import smri_mae_coarse2fine.model_mae as c2f_models
import smri_mae_edgeloss.main_pretrain as mp
import smri_mae_edgeloss.model_mae as edge_models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_SEED = 7338
BOUNDARY_PERCENTILE = 0.80  # top 20% by gradient magnitude = "boundary"


def gradient_magnitude_3d(x: torch.Tensor) -> torch.Tensor:
    """Same formula as smri_mae_edgeloss.model_mae.gradient_magnitude_3d."""
    x = F.pad(x, (1, 1, 1, 1, 1, 1), mode="replicate")
    gd = x[:, :, 2:, 1:-1, 1:-1] - x[:, :, :-2, 1:-1, 1:-1]
    gh = x[:, :, 1:-1, 2:, 1:-1] - x[:, :, 1:-1, :-2, 1:-1]
    gw = x[:, :, 1:-1, 1:-1, 2:] - x[:, :, 1:-1, 1:-1, :-2]
    return torch.sqrt(gd**2 + gh**2 + gw**2 + 1e-8)


def load_model(models_module, ckpt_path: Path, config_path: Path, edge_loss_weight_override=None):
    args = OmegaConf.load(config_path)
    if edge_loss_weight_override is not None and "edge_loss_weight" in (args.get("model_kwargs") or {}):
        args.model_kwargs.edge_loss_weight = edge_loss_weight_override
    model = models_module.MODELS_DICT[args.model](
        img_size=args.img_size,
        in_chans=args.get("in_chans", 1),
        patch_size=args.patch_size,
        **(args.get("model_kwargs") or {}),
    ) if hasattr(models_module, "MODELS_DICT") else getattr(models_module, args.model)(
        img_size=args.img_size,
        in_chans=args.get("in_chans", 1),
        patch_size=args.patch_size,
        **(args.get("model_kwargs") or {}),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()
    return model, args


print("loading val data (edgeloss's config, shared val cache)...")
edge_args = OmegaConf.load("checkpoints/experiments/vitl_edgeloss_4090/config.yaml")
_, eval_loaders = mp.create_data_loaders(edge_args)
val_loader = eval_loaders["fomo_val"]

print("loading baseline model...")
baseline_model, baseline_args = load_model(
    baseline_models, Path("/workspace/smri-fm/checkpoints/experiments/vitl_baseline_4090/checkpoint-00099.pth"),
    Path("/workspace/smri-fm/checkpoints/experiments/vitl_baseline_4090/config.yaml"),
)
print("loading coarse2fine model...")
c2f_model, c2f_args = load_model(
    c2f_models, Path("/workspace/smri-fm-coarse2fine/checkpoints/experiments/vitl_coarse2fine_4090/checkpoint-00099.pth"),
    Path("/workspace/smri-fm-coarse2fine/checkpoints/experiments/vitl_coarse2fine_4090/config.yaml"),
)
print("loading edgeloss model...")
edge_model, _ = load_model(
    edge_models, Path("checkpoints/experiments/vitl_edgeloss_4090/checkpoint-00099.pth"),
    Path("checkpoints/experiments/vitl_edgeloss_4090/config.yaml"),
)

models = {"baseline": baseline_model, "coarse2fine": c2f_model, "edgeloss": edge_model}
totals = {name: {"boundary_se": 0.0, "boundary_n": 0.0, "interior_se": 0.0, "interior_n": 0.0} for name in models}

amp_dtype = torch.bfloat16
num_batches = len(val_loader)
print(f"running {num_batches} val batches through all three models...")

with torch.inference_mode():
    for batch_idx, batch in enumerate(val_loader):
        images, img_mask = mri_data.densify_sparse_image_batch(
            batch["image_values"], batch["img_mask"], (1, *edge_args.img_size), dtype=amp_dtype,
        )
        images, img_mask = images.to(DEVICE), img_mask.to(DEVICE)
        grad_mag = gradient_magnitude_3d(images.float())

        for name, model in models.items():
            torch.manual_seed(EVAL_SEED + batch_idx)
            torch.cuda.manual_seed(EVAL_SEED + batch_idx)
            with torch.autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=True):
                loss, state = model(
                    images, img_mask=img_mask, mask_ratio=0.8, pred_mask_ratio=None,
                    pad_to_multiple=32, with_state=True,
                )
            pred_images = state["pred_images"].float()
            pred_mask = state["pred_mask"]  # dense bool, voxels that were masked+predicted
            scored = pred_mask & img_mask

            grad_at_scored = grad_mag[scored]
            if grad_at_scored.numel() == 0:
                continue
            thresh = torch.quantile(grad_at_scored.float(), BOUNDARY_PERCENTILE)
            boundary_mask = scored & (grad_mag >= thresh)
            interior_mask = scored & (grad_mag < thresh)

            sq_err = (pred_images - images.float()) ** 2
            totals[name]["boundary_se"] += sq_err[boundary_mask].sum().item()
            totals[name]["boundary_n"] += boundary_mask.sum().item()
            totals[name]["interior_se"] += sq_err[interior_mask].sum().item()
            totals[name]["interior_n"] += interior_mask.sum().item()

        if batch_idx % 20 == 0:
            print(f"  batch {batch_idx}/{num_batches}")

print()
print(f"{'model':<14}{'boundary MSE':<16}{'interior MSE':<16}")
for name, t in totals.items():
    b_mse = t["boundary_se"] / max(t["boundary_n"], 1)
    i_mse = t["interior_se"] / max(t["interior_n"], 1)
    print(f"{name:<14}{b_mse:<16.4f}{i_mse:<16.4f}")
