import torch
from schedule import T


@torch.no_grad()
def sample(unet, labels, betas, alphas, alpha_bars, record=(), size=(1, 28, 28)):
    """labels: (N,) long tensor of class ids, on the same device as the unet.
    record: step indices whose INPUT x_t is saved (e.g. 999, 750, ...). Index 0 saves the
    final clean sample. Returns (samples in [-1,1], {t: x_t})."""
    unet.eval()
    dev = labels.device
    n = len(labels)
    x = torch.randn(n, *size, device=dev)                  # x_T ~ N(0, I)
    traj = {}
    for t in reversed(range(T)):                           # t = 999, ..., 0
        if t in record and t != 0:
            traj[t] = x.clone()
        tt = torch.full((n,), t, device=dev, dtype=torch.long)
        eps = unet(x, tt, class_labels=labels).sample
        mean = (x - betas[t] / (1.0 - alpha_bars[t]).sqrt() * eps) / alphas[t].sqrt()
        x = mean + betas[t].sqrt() * torch.randn_like(x) if t > 0 else mean
    x = x.clamp(-1, 1)
    if 0 in record:
        traj[0] = x.clone()
    return x, traj