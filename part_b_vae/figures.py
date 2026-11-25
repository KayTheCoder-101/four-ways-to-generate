"""Part B figures (B.4): reconstructions, 64 samples from N(0,I), interpolation.
Runs fine on CPU. Needs part_b_vae/vae.pt (from train.py).

Usage (from part_b_vae/):
    python figures.py
"""
import argparse, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "starter"))
from uhat import load_uhat_characters
from viz import save_grid
from vae import VAE

p = argparse.ArgumentParser()
p.add_argument("--data-root", default=os.path.join(HERE, "..", "starter", "data", "uhat"))
p.add_argument("--weights", default=os.path.join(HERE, "vae.pt"))
p.add_argument("--out", default=os.path.join(HERE, "..", "results"))
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)

ckpt = torch.load(args.weights, map_location="cpu")
names = ckpt["class_names"]
model = VAE(d=16)
model.load_state_dict(ckpt["model"])
model.eval()

x_te, y_te, _ = load_uhat_characters(args.data_root, "test", class_names=names)

with torch.no_grad():
    # (a) 16 test images and their reconstructions (decode the posterior mean)
    idx = torch.randperm(len(x_te))[:16]
    x = x_te[idx]
    mu, _ = model.encode(x)
    recon = torch.sigmoid(model.decode(mu))
    # layout: rows = originals(8), recon(8), originals(8), recon(8)
    grid = torch.cat([x[:8], recon[:8], x[8:], recon[8:]])
    save_grid(grid, os.path.join(args.out, "vae_reconstructions.png"), nrow=8)

    # (b) 64 new characters decoded from z ~ N(0, I)
    samples = model.sample(64)
    save_grid(samples, os.path.join(args.out, "vae_samples.png"), nrow=8)

    # (c) interpolation between mu of two test images of DIFFERENT characters
    n_pairs, steps = 6, 10
    rows, titles = [], []
    g = torch.Generator().manual_seed(args.seed)
    for _ in range(n_pairs):
        while True:
            i, j = torch.randint(len(x_te), (2,), generator=g).tolist()
            if y_te[i] != y_te[j]:
                break
        m1, _ = model.encode(x_te[i:i + 1])
        m2, _ = model.encode(x_te[j:j + 1])
        alphas = torch.linspace(0, 1, steps).view(-1, 1)
        z = (1 - alphas) * m1 + alphas * m2           # (10, 16) points on the line
        rows.append(torch.sigmoid(model.decode(z)))
        t = [""] * steps
        t[0], t[-1] = names[y_te[i]], names[y_te[j]]
        titles += t
    save_grid(torch.cat(rows), os.path.join(args.out, "vae_interpolation.png"),
              nrow=steps, titles=titles)

print("saved vae_reconstructions.png, vae_samples.png, vae_interpolation.png in", args.out)