"""
Fair-comparison eval: loads the trained edgeloss checkpoint but scores it with
edge_loss_weight=0, so the reported loss falls back to the exact same plain
unweighted per-voxel MSE formula that the baseline and coarse2fine's fine_loss
use. The edge weighting is a loss-time setting only (self.edge_loss_weight is
a plain python float, not a learned parameter) -- the checkpoint's actual
weights are untouched, only how the loss is *scored* changes here.

Reuses main_pretrain.py's own create_data_loaders/evaluate unchanged, so this
runs the exact same eval code path (same masking, same eval_seed, same val
data) that produced the baseline/coarse2fine numbers -- just with one setting
flipped.
"""
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent / "src"))

import smri_mae_edgeloss.main_pretrain as mp
import smri_mae_edgeloss.utils as ut

CKPT_DIR = Path("checkpoints/experiments/vitl_edgeloss_4090")

args = OmegaConf.load(CKPT_DIR / "config.yaml")
args.model_kwargs.edge_loss_weight = 0  # the only change: score with plain MSE
args.wandb = False

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

print("building data loaders (val only needed, but reusing create_data_loaders as-is)...")
_, eval_loaders = mp.create_data_loaders(args)

print("building model with edge_loss_weight=0...")
model = mp.MODELS_DICT[args.model](
    img_size=args.img_size,
    in_chans=args.get("in_chans", 1),
    patch_size=args.patch_size,
    **(args.get("model_kwargs") or {}),
)
model.to(device)

ckpt_path = CKPT_DIR / "checkpoint-00099.pth"
print(f"loading trained weights from {ckpt_path} (weights unchanged, only the loss-time setting above differs)...")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["model"])

stats, _ = mp.evaluate(
    args, model, eval_loaders["fomo_val"], epoch=ckpt["epoch"], device=device, eval_name="fomo_val"
)
print()
print("=== plain unweighted MSE (fair comparison to baseline / coarse2fine fine_loss) ===")
print(stats)
