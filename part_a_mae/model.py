"""Part A - Masked autoencoder: model, masking, loss and correctness checks."""
import os
import sys
import torch
import torch.nn as nn

# this file lives in part_a_mae/, so the repo root is one level up
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(ROOT,"starter"))

from eurosat import patchify

RESULTS_DIR=os.path.join(ROOT,"results")
FIG_DIR=os.path.join(RESULTS_DIR,"figures")
DEFAULT_CKPT_DIR=os.path.join(ROOT,"checkpoints","mae")


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_trained_mae(ckpt_dir,device):
    """Load the finished pretrained MAE (mae_final.pt) from ckpt_dir."""
    path=os.path.join(ckpt_dir,"mae_final.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run pretrain.py first")
    model=MAE().to(device)
    model.load_state_dict(torch.load(path,map_location=device))
    model.eval()
    return model


#-----------Model (Steps 1-7)
class MAE(nn.Module):
    #-----------Step 1: __init__, creating all the layers
    def __init__(self):
        super().__init__()

        #encoder side
        self.patch_embed=nn.Linear(192,192)
        self.pos_embed=nn.Parameter(torch.randn(1,64,192)*0.02)
        enc_layer=nn.TransformerEncoderLayer(d_model=192,nhead=3,dim_feedforward=768,batch_first=True)
        self.encoder=nn.TransformerEncoder(enc_layer,num_layers=6,enable_nested_tensor=False)
        self.enc_norm=nn.LayerNorm(192)

        #decoder side
        self.dec_embed=nn.Linear(192,128)
        self.mask_token=nn.Parameter(torch.randn(1,1,128)*0.02)
        self.dec_pos_embed=nn.Parameter(torch.randn(1,64,128)*0.02)
        self.dec_layers=nn.ModuleList([nn.TransformerEncoderLayer(d_model=128,nhead=4,batch_first=True) for _ in range(2)])
        self.dec_pred=nn.Linear(128,192)

    #-----------Step 2: random_masking(x), input [B, 64, D]
    def random_masking(self,x):
        B,N,D=x.shape

        noise=torch.rand(B,N,device=x.device)
        shuffle_idx=torch.argsort(noise,dim=1)
        restore_idx=torch.argsort(shuffle_idx,dim=1)

        keep_idx=shuffle_idx[:,:16]
        x_visible=torch.gather(x,1,keep_idx.unsqueeze(-1).expand(-1,-1,D))

        #1 means hidden, reordered to line up with the original patch positions
        mask=torch.ones(B,N,device=x.device)
        mask[:,:16]=0
        mask=torch.gather(mask,1,restore_idx)

        return x_visible,mask,restore_idx

    #-----------Step 3: encode(patches), input [B, 64, 192]
    def encode(self,patches):
        x=self.patch_embed(patches)+self.pos_embed
        x_vis,mask,restore_idx=self.random_masking(x)
        x_vis=self.enc_norm(self.encoder(x_vis))
        return x_vis,mask,restore_idx

    #-----------Step 4: decode(x_vis, restore_idx)
    def decode(self,x_vis,restore_idx):
        x=self.dec_embed(x_vis)
        B=x.shape[0]

        #visible first, then 48 mask tokens (still shuffled)
        masks=self.mask_token.expand(B,48,-1)
        x=torch.cat([x,masks],dim=1)

        #unshuffle so every token is in its real position
        x=torch.gather(x,1,restore_idx.unsqueeze(-1).expand(-1,-1,x.shape[2]))
        x=x+self.dec_pos_embed

        for layer in self.dec_layers:
            x=layer(x)

        pred=self.dec_pred(x)
        return pred

    #-----------Step 5: loss(pred, target, mask)
    def loss(self,pred,target,mask):
        per_patch=((pred-target)**2).mean(dim=-1)
        return (per_patch*mask).sum()/mask.sum()

    #-----------Step 6: forward(images)
    def forward(self,images):
        patches=patchify(images)
        x_vis,mask,restore_idx=self.encode(patches)
        pred=self.decode(x_vis,restore_idx)
        loss=self.loss(pred,patches,mask)
        return loss,pred,mask

    #-----------Step 7: features(images), for the classifier later
    def features(self,images):
        patches=patchify(images)
        x=self.patch_embed(patches)+self.pos_embed
        x=self.enc_norm(self.encoder(x))
        return x.mean(dim=1)


def sanity_check(model,images):
    model.eval()
    with torch.no_grad():
        patches=patchify(images)
        print("patches:",tuple(patches.shape))

        x_vis,mask,restore_idx=model.encode(patches)
        print("x_vis:",tuple(x_vis.shape))
        print("mask:",tuple(mask.shape),"| restore_idx:",tuple(restore_idx.shape))

        pred=model.decode(x_vis,restore_idx)
        print("pred:",tuple(pred.shape))

        loss=model.loss(pred,patches,mask)
        print("loss:",round(loss.item(),4))

        feats=model.features(images)
        print("features:",tuple(feats.shape))

        #check 1: exactly 48 of 64 patches are masked in every image
        masked_counts=mask.sum(dim=1)
        assert (masked_counts==48).all(),"every image must have 48 masked patches"
        print("CHECK masked patches per image:",masked_counts.unique().tolist(),"(expected [48.0])")

        #check 2: loss only depends on masked patches
        #replace predictions at VISIBLE positions with garbage; loss must not change
        pred_garbage=torch.where(mask.unsqueeze(-1).bool(),pred,torch.randn_like(pred)*100)
        loss_garbage=model.loss(pred_garbage,patches,mask)
        assert torch.isclose(loss,loss_garbage),"loss changed when visible predictions changed"
        print(f"CHECK loss only on masked patches: {loss.item():.6f} == {loss_garbage.item():.6f}")

    model.train()
    print("Sanity check passed")


if __name__=="__main__":
    #quick check on random images: shapes + the two required correctness checks
    torch.manual_seed(0)
    sanity_check(MAE(),torch.rand(8,3,64,64))
