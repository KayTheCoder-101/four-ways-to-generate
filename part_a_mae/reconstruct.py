"""Part A - A.2: masked input / reconstruction / original for 8 test images."""
import os
import argparse
import torch

from model import load_trained_mae,get_device,FIG_DIR,DEFAULT_CKPT_DIR
from eurosat import load_eurosat_split,patchify,unpatchify
from viz import save_grid


def reconstruct(model,x_test,device,fig_dir):
    model.eval()
    images=x_test[:8].to(device)
    with torch.no_grad():
        _,pred,mask=model(images)

    orig_patches=patchify(images)

    #mask as [B, 64, 1] so it broadcasts over the 192 numbers per patch
    m=mask.unsqueeze(-1).bool()

    grey=torch.full_like(orig_patches,0.5)
    masked_input=unpatchify(torch.where(m,grey,orig_patches))
    recon=unpatchify(torch.where(m,pred,orig_patches))

    rows=torch.cat([masked_input,recon,images],dim=0)
    save_grid(rows,os.path.join(fig_dir,"mae_reconstructions.png"),nrow=8)
    print("Saved mae_reconstructions.png (rows: masked input, reconstruction, original)")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir",default=DEFAULT_CKPT_DIR)
    args=parser.parse_args()

    torch.manual_seed(42)
    os.makedirs(FIG_DIR,exist_ok=True)
    device=get_device()

    x_test,_=load_eurosat_split("test")
    model=load_trained_mae(args.ckpt_dir,device)
    reconstruct(model,x_test,device,FIG_DIR)
