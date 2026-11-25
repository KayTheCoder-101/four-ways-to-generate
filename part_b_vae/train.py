"""Part B: train the VAE on UHaT characters.
Adam, lr 1e-3, batch 128, 30 epochs. Logs recon / KL / total (nats per image)
for train and test every epoch.

Usage (from part_b_vae/):
    python train.py --data-root ../starter/data/uhat
"""
import argparse, json, os, sys, time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "starter"))
from uhat import load_uhat_characters      # starter code, unchanged
from vae import VAE, vae_loss

p = argparse.ArgumentParser()
p.add_argument("--data-root", default=os.path.join(HERE, "..", "starter", "data", "uhat"))
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--out", default=os.path.join(HERE, "..", "results"))
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
print("device:", dev, gpu)

x_tr, _, names = load_uhat_characters(args.data_root, "train")
x_te, _, _ = load_uhat_characters(args.data_root, "test", class_names=names)
x_tr, x_te = x_tr.to(dev), x_te.to(dev)       # labels are NOT used in Part B
print("train", tuple(x_tr.shape), "test", tuple(x_te.shape), "classes", len(names))

model = VAE(d=16).to(dev)
n_params = sum(p.numel() for p in model.parameters())
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
print("parameters:", n_params)


@torch.no_grad()
def evaluate(x):
    model.eval()
    tot = rec = kl = 0.0
    for i in range(0, len(x), 512):
        xb = x[i:i + 512]
        logits, mu, logvar = model(xb)
        t, r, k = vae_loss(logits, xb, mu, logvar)
        n = len(xb)
        tot += t.item() * n; rec += r.item() * n; kl += k.item() * n
    n = len(x)
    return tot / n, rec / n, kl / n


hist = {"train": [], "test": []}
t0 = time.time()
for ep in range(1, args.epochs + 1):
    model.train()
    perm = torch.randperm(len(x_tr), device=dev)
    tot = rec = kl = 0.0
    for i in range(0, len(x_tr), args.bs):
        xb = x_tr[perm[i:i + args.bs]]
        logits, mu, logvar = model(xb)
        loss, r, k = vae_loss(logits, xb, mu, logvar)
        opt.zero_grad(); loss.backward(); opt.step()
        n = len(xb)
        tot += loss.item() * n; rec += r.item() * n; kl += k.item() * n
    n = len(x_tr)
    tr = (tot / n, rec / n, kl / n)
    te = evaluate(x_te)
    hist["train"].append(tr); hist["test"].append(te)
    print(f"ep {ep:2d} | train total {tr[0]:.2f} rec {tr[1]:.2f} kl {tr[2]:.2f} "
          f"| test total {te[0]:.2f} rec {te[1]:.2f} kl {te[2]:.2f}")

elapsed = time.time() - t0
print(f"training time: {elapsed/60:.1f} min on {gpu}")

torch.save({"model": model.state_dict(), "class_names": names}, os.path.join(HERE, "vae.pt"))
summary = {"params": n_params, "final_train_loss": hist["train"][-1][0],
           "test_total": hist["test"][-1][0], "test_recon": hist["test"][-1][1],
           "test_kl": hist["test"][-1][2], "time_sec": elapsed, "device": gpu,
           "history": hist}
with open(os.path.join(args.out, "vae_history.json"), "w") as f:
    json.dump(summary, f, indent=1)

# loss-curve figure: reconstruction, KL and total, train vs test
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
for ax, j, title in zip(axes, [1, 2, 0], ["Reconstruction (BCE)", "KL", "Negative ELBO"]):
    ax.plot([h[j] for h in hist["train"]], label="train")
    ax.plot([h[j] for h in hist["test"]], label="test")
    ax.set_title(title); ax.set_xlabel("epoch"); ax.set_ylabel("nats / image"); ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(args.out, "vae_loss_terms.png"), dpi=150)
print("saved vae.pt, results/vae_history.json, results/vae_loss_terms.png")