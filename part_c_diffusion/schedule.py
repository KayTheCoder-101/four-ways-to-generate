import torch

T = 1000
BETA_START, BETA_END = 1e-4, 0.02


def make_schedule(T=T, beta_start=BETA_START, beta_end=BETA_END, device="cpu"):
    """Returns betas, alphas, alpha_bars (all shape (T,)).
    beta_t linear, alpha_t = 1 - beta_t, alpha_bar_t = prod_{s<=t} alpha_s."""
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)       # float64 for accuracy
    return (betas.float().to(device), alphas.float().to(device),
            alpha_bars.float().to(device))


def q_sample(x0, t, alpha_bars, noise=None):
    """Forward process in closed form:
    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps,  eps ~ N(0, I).
    x0: (B, C, H, W) in [-1, 1]; t: (B,) long. Returns (x_t, eps)."""
    if noise is None:
        noise = torch.randn_like(x0)
    ab = alpha_bars[t].view(-1, 1, 1, 1)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise, noise