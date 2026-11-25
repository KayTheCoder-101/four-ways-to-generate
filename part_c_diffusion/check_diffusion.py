"""Part C correctness checks (section 3.6) + forward-noising figure (C.2).
Runs on CPU. Paste the printed output into the README.

Usage (from part_c_diffusion/):
    python check_diffusion.py
"""
import argparse, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "starter"))
from uhat import load_uhat_characters
from viz import save_grid
from schedule import make_schedule, q_sample, T

p = argparse.ArgumentParser()
p.add_argument("--data-root", default=os.path.join(HERE, "..", "starter", "data", "uhat"))
p.add_argument("--out", default=os.path.join(HERE, "..", "results"))
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(0)

betas, alphas, alpha_bars = make_schedule()
print(f"betas: {betas[0].item():.4g} -> {betas[-1].item():.4g} (len {len(betas)})")
print(f"alpha_bar at t=0: {alpha_bars[0].item():.6f}")
print(f"alpha_bar_T (t=999): {alpha_bars[-1].item():.3e}   (expected about 4e-5)")
assert 3e-5 < alpha_bars[-1].item() < 5e-5
assert (alpha_bars[1:] < alpha_bars[:-1]).all(), "alpha_bar must be decreasing"

# data: UHaT characters rescaled from [0,1] to [-1,1]
x_tr, y_tr, names = load_uhat_characters(args.data_root, "train")
x_tr = x_tr * 2 - 1
print("train", tuple(x_tr.shape), "range", x_tr.min().item(), x_tr.max().item(),
      "| classes", len(names))

# forward process on one image at t in {0,100,250,500,750,999}
ts = [0, 100, 250, 500, 750, 999]
x0 = x_tr[0:1].repeat(len(ts), 1, 1, 1)
eps = torch.randn_like(x0[:1]).repeat(len(ts), 1, 1, 1)    # same noise for every t
xt, _ = q_sample(x0, torch.tensor(ts), alpha_bars, noise=eps)
save_grid((xt.clamp(-1, 1) + 1) / 2, os.path.join(args.out, "ddpm_forward.png"),
          nrow=len(ts), titles=[f"t={t}" for t in ts])
print("image class:", names[y_tr[0]], "-> saved results/ddpm_forward.png")

# the noised image at t=999 should look like pure noise: unit variance, no link to x0
x0b = x_tr[:512]
xt999, _ = q_sample(x0b, torch.full((512,), 999), alpha_bars)
corr = torch.corrcoef(torch.stack([x0b.flatten(), xt999.flatten()]))[0, 1].item()
print(f"x_999: mean {xt999.mean().item():.3f}, std {xt999.std().item():.3f}, "
      f"corr with x0 {corr:.3f}  (noise-like: mean~0, std~1, corr~0)")
assert abs(xt999.std().item() - 1.0) < 0.1 and abs(corr) < 0.05

# U-Net constructor from the assignment: random init, shape + parameter count
from diffusers import UNet2DModel
unet = UNet2DModel(
    sample_size=28, in_channels=1, out_channels=1, layers_per_block=2,
    block_out_channels=(32, 64, 128),
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    num_class_embeds=40)
n_params = sum(p.numel() for p in unet.parameters())
with torch.no_grad():
    out = unet(xt[:4], torch.tensor(ts[:4]), class_labels=y_tr[:4]).sample
print("U-Net parameters:", n_params, "| output shape", tuple(out.shape))
assert out.shape == (4, 1, 28, 28)
print("ALL DIFFUSION CHECKS PASSED")