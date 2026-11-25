"""Part C figures and blind-test material (C.5, section 3.3).
Needs part_c_diffusion/ddpm.pt. Use a GPU (Colab): it runs 1000 sampling steps
for 320 + 120 images.

Usage (from part_c_diffusion/):
    python figures.py

Outputs (in ../results unless noted):
    ddpm_samples_grid.png     40 characters x 8 samples, name on every row
    ddpm_denoising.png        denoising of 3 samples at t = 999,750,500,250,100,0
    ddpm_blind_test.png       40 shuffled samples, numbered 1..40, NO labels
    human_eval_diffusion.csv  empty sheet for the two members' guesses
    blind_key.csv             (in part_c_diffusion/) the answer key. DO NOT OPEN
                              or commit it until both members have guessed.
"""
import argparse, csv, os, sys
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers import UNet2DModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "starter"))
from viz import save_grid
from schedule import make_schedule
from sampler import sample

p = argparse.ArgumentParser()
p.add_argument("--weights", default=os.path.join(HERE, "ddpm.pt"))
p.add_argument("--out", default=os.path.join(HERE, "..", "results"))
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)
ckpt = torch.load(args.weights, map_location=dev)
names = ckpt["class_names"]
unet = UNet2DModel(
    sample_size=28, in_channels=1, out_channels=1, layers_per_block=2,
    block_out_channels=(32, 64, 128),
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    num_class_embeds=40).to(dev)
unet.load_state_dict(ckpt["model"])
unet.eval()
betas, alphas, alpha_bars = make_schedule(device=dev)
to01 = lambda x: ((x + 1) / 2).clamp(0, 1).cpu()


# ---- 1. 40 x 8 grid: 8 samples for each of the 40 characters ----
torch.manual_seed(args.seed)
labels = torch.arange(40, device=dev).repeat_interleave(8)
xs, _ = sample(unet, labels, betas, alphas, alpha_bars)
imgs = to01(xs).view(40, 8, 28, 28)


def draw_panel(ax, block, class_names):
    rows = []
    for c in range(block.shape[0]):
        parts = []
        for k in range(block.shape[1]):
            parts += [block[c, k], torch.full((28, 2), 0.35)]
        rows += [torch.cat(parts[:-1], 1), torch.full((2, 8 * 30 - 2), 0.35)]
    ax.imshow(torch.cat(rows[:-1], 0), cmap="gray", vmin=0, vmax=1)
    ax.set_yticks([c * 30 + 14 for c in range(block.shape[0])])
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)


fig, axes = plt.subplots(1, 2, figsize=(9, 10))
draw_panel(axes[0], imgs[:20], names[:20])
draw_panel(axes[1], imgs[20:], names[20:])
fig.tight_layout()
fig.savefig(os.path.join(args.out, "ddpm_samples_grid.png"), dpi=150)
plt.close(fig)
print("saved ddpm_samples_grid.png")

# ---- 2. denoising trajectory of 3 samples ----
ts = [999, 750, 500, 250, 100, 0]
torch.manual_seed(args.seed + 1)
pick = torch.randperm(40)[:3].tolist()
lab = torch.tensor(pick, device=dev)
_, traj = sample(unet, lab, betas, alphas, alpha_bars, record=ts)
grid, titles = [], []
for i, c in enumerate(pick):
    for t in ts:
        grid.append(to01(traj[t][i]))
        titles.append(f"{names[c]}  t={t}")
save_grid(torch.stack(grid), os.path.join(args.out, "ddpm_denoising.png"),
          nrow=len(ts), titles=titles)
print("saved ddpm_denoising.png")

# ---- 3. blind recognisability test: one sample per character, shuffled ----
torch.manual_seed(args.seed + 2)
xs, _ = sample(unet, torch.arange(40, device=dev), betas, alphas, alpha_bars)
perm = torch.randperm(40, generator=torch.Generator().manual_seed(args.seed + 3))
shuffled = to01(xs)[perm]
save_grid(shuffled, os.path.join(args.out, "ddpm_blind_test.png"), nrow=8,
          titles=[str(i + 1) for i in range(40)])

with open(os.path.join(args.out, "human_eval_diffusion.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image_id", "member1_guess", "member2_guess"])
    for i in range(40):
        w.writerow([i + 1, "", ""])
with open(os.path.join(HERE, "blind_key.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image_id", "true_label"])
    for i in range(40):
        w.writerow([i + 1, names[perm[i].item()]])
print("saved ddpm_blind_test.png, human_eval_diffusion.csv, blind_key.csv (do not open the key yet)")
print("class names to choose from:", names)