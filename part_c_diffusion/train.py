import argparse, json, os, sys, time
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers import UNet2DModel          # architecture only, random init

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "starter"))
from uhat import load_uhat_characters
from viz import save_grid
from schedule import make_schedule, q_sample, T
from sampler import sample

p = argparse.ArgumentParser()
p.add_argument("--data-root", default=os.path.join(HERE, "..", "starter", "data", "uhat"))
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--out", default=os.path.join(HERE, "..", "results"))
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
print("device:", dev, gpu)

x_tr, y_tr, names = load_uhat_characters(args.data_root, "train")
x_tr = (x_tr * 2 - 1).to(dev)                      # rescale [0,1] -> [-1,1]
y_tr = y_tr.to(dev)
print("train", tuple(x_tr.shape), "classes", len(names))

betas, alphas, alpha_bars = make_schedule(device=dev)

unet = UNet2DModel(
    sample_size=28, in_channels=1, out_channels=1, layers_per_block=2,
    block_out_channels=(32, 64, 128),
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    num_class_embeds=40).to(dev)
n_params = sum(p.numel() for p in unet.parameters())
print("U-Net parameters:", n_params)
opt = torch.optim.Adam(unet.parameters(), lr=args.lr)

CKPT = os.path.join(HERE, "ddpm.pt")
losses = []
t0 = time.time()
for ep in range(1, args.epochs + 1):
    unet.train()
    perm = torch.randperm(len(x_tr), device=dev)
    tot, n = 0.0, 0
    for i in range(0, len(x_tr), args.bs):
        idx = perm[i:i + args.bs]
        x0, y = x_tr[idx], y_tr[idx]
        t = torch.randint(0, T, (len(idx),), device=dev)       # t ~ Uniform over T steps
        xt, eps = q_sample(x0, t, alpha_bars)                   # eps ~ N(0, I)
        pred = unet(xt, t, class_labels=y).sample
        loss = F.mse_loss(pred, eps)                            # L_simple
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(idx); n += len(idx)
    losses.append(tot / n)
    print(f"epoch {ep:2d}/{args.epochs} | loss {losses[-1]:.4f} | {(time.time()-t0)/60:.1f} min")
    if ep % 10 == 0:                                            # protect against Colab disconnects
        torch.save({"model": unet.state_dict(), "class_names": names, "epoch": ep}, CKPT)

elapsed = time.time() - t0
print(f"training time: {elapsed/60:.1f} min on {gpu}")
torch.save({"model": unet.state_dict(), "class_names": names, "epoch": args.epochs}, CKPT)

with open(os.path.join(args.out, "ddpm_history.json"), "w") as f:
    json.dump({"params": n_params, "final_train_loss": losses[-1], "time_sec": elapsed,
               "device": gpu, "losses": losses}, f, indent=1)

plt.figure(figsize=(5, 3.5))
plt.plot(range(1, len(losses) + 1), losses)
plt.xlabel("epoch"); plt.ylabel("L_simple (MSE on noise)"); plt.title("DDPM training loss")
plt.tight_layout(); plt.savefig(os.path.join(args.out, "ddpm_loss.png"), dpi=150)

# quick sanity check: one sample per class (40 images)
labels = torch.arange(40, device=dev)
xs, _ = sample(unet, labels, betas, alphas, alpha_bars)
save_grid((xs + 1) / 2, os.path.join(args.out, "ddpm_quick_samples.png"), nrow=8,
          titles=names)
print("saved ddpm.pt, ddpm_history.json, ddpm_loss.png, ddpm_quick_samples.png")