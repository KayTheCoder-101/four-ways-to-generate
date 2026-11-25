# What the app expects from Parts B and C

The app loads each model by file path, so these names must match exactly.

## Part B — `part_b_vae/model.py`
- `class VAE(nn.Module)` with no required constructor arguments
- `encode(x) -> (mu, logvar)`  — `x` is `[N, 1, 28, 28]` in `[0, 1]`, both outputs `[N, 16]`
- `decode(z) -> logits`        — `z` is `[N, 16]`, output `[N, 1, 28, 28]` **logits** (the app applies sigmoid)
- weights: `torch.save(model.state_dict(), "vae_final.pt")`

## Part C — `part_c_diffusion/ddpm.py`
- `build_unet()` → the `UNet2DModel` with the exact constructor from the manual
- `sample(unet, labels, device)` → images `[N, 1, 28, 28]` in `[-1, 1]`
  (`labels` is a LongTensor of class indices, same order as the UHaT loader's `class_names`)
- weights: `torch.save(unet.state_dict(), "ddpm_final.pt")`

Upload `vae_final.pt` and `ddpm_final.pt` to the Hugging Face repo (or put them in `checkpoints/`),
and the VAE and Diffusion tabs start working with no change to `app.py`.
