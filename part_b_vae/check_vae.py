import torch
from vae import VAE, kl_divergence, vae_loss

torch.manual_seed(0)
model = VAE(d=16)
x = torch.rand(8, 1, 28, 28)

logits, mu, logvar = model(x)
print("input", tuple(x.shape), "-> logits", tuple(logits.shape),
      "| mu", tuple(mu.shape), "| logvar", tuple(logvar.shape))
assert logits.shape == x.shape

print("VAE parameters:", sum(p.numel() for p in model.parameters()))

# KL is never negative (random mu / logvar)
kl = kl_divergence(torch.randn(1000, 16) * 3, torch.randn(1000, 16))
print("KL min over 1000 random posteriors:", kl.min().item())
assert (kl >= 0).all()

# KL is 0 when mu = 0 and logvar = 0
kl0 = kl_divergence(torch.zeros(4, 16), torch.zeros(4, 16))
print("KL at mu=0, logvar=0:", kl0.tolist())
assert torch.allclose(kl0, torch.zeros(4))

total, rec, k = vae_loss(logits, x, mu, logvar)
print(f"loss check: total={total.item():.2f} recon={rec.item():.2f} kl={k.item():.2f}")
assert torch.isclose(total, rec + k)

print("sample shape:", tuple(model.sample(64).shape))
print("ALL VAE CHECKS PASSED")