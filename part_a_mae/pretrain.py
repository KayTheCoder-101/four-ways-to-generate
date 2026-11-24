"""Part A - A.1: pretrain the MAE on the 16.2k EuroSAT training images (labels ignored).
Resumable: re-run with the same --ckpt_dir after a Colab disconnect."""
import os
import json
import time
import argparse
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import MAE,sanity_check,get_device,RESULTS_DIR,FIG_DIR,DEFAULT_CKPT_DIR
from eurosat import load_eurosat_split


def latest_checkpoint(ckpt_dir):
    files=[f for f in os.listdir(ckpt_dir) if f.startswith("mae_epoch") and f.endswith(".pt")]
    if not files:
        return None
    files.sort(key=lambda f:int(f[len("mae_epoch"):-3]))
    return os.path.join(ckpt_dir,files[-1])


def pretrain(model,x_train,device,ckpt_dir,fig_dir,results_dir,epochs=100):
    batch_size=256
    save_every=10

    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=0.05)
    n_params=sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params}")

    losses=[]
    start_epoch=1
    prev_minutes=0.0

    #resume from the newest epoch checkpoint if there is one
    ckpt_path=latest_checkpoint(ckpt_dir)
    if ckpt_path is not None:
        ckpt=torch.load(ckpt_path,map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        losses=ckpt["losses"]
        start_epoch=ckpt["epoch"]+1
        prev_minutes=ckpt.get("minutes",0.0)
        print(f"Resuming from {os.path.basename(ckpt_path)} (epoch {ckpt['epoch']})")

    model.train()
    t0=time.time()

    for epoch in range(start_epoch,epochs+1):
        perm=torch.randperm(len(x_train))
        total=0.0
        n_batches=0

        for start in range(0,len(x_train),batch_size):
            idx=perm[start:start+batch_size]
            batch=x_train[idx].to(device)

            loss,_,_=model(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total+=loss.item()
            n_batches+=1

        avg_loss=total/n_batches
        losses.append(avg_loss)
        elapsed_minutes=prev_minutes+(time.time()-t0)/60.0
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Time: {elapsed_minutes:.2f} min")

        #colab will disconnect at some point
        if epoch%save_every==0:
            torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                        "epoch":epoch,"losses":losses,"minutes":elapsed_minutes},
                       os.path.join(ckpt_dir,f"mae_epoch{epoch}.pt"))

    torch.save(model.state_dict(),os.path.join(ckpt_dir,"mae_final.pt"))

    plt.figure(figsize=(8,5))
    plt.plot(range(1,len(losses)+1),losses,label="Pretraining Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Masked MSE")
    plt.title("MAE Pretraining Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir,"mae_pretrain_loss.png"),bbox_inches="tight")
    plt.close()

    #numbers for Table 2
    minutes=prev_minutes+(time.time()-t0)/60.0
    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    table2={"parameters":n_params,"final_loss":losses[-1],"minutes":round(minutes,2),"gpu":gpu}
    with open(os.path.join(results_dir,"mae_pretrain_log.json"),"w") as f:
        json.dump({"table2":table2,"losses":losses},f,indent=4)
    print(table2)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    #on colab, point this at Google Drive so checkpoints survive a disconnect
    parser.add_argument("--ckpt_dir",default=DEFAULT_CKPT_DIR)
    parser.add_argument("--epochs",type=int,default=100)
    args=parser.parse_args()

    torch.manual_seed(42)
    os.makedirs(args.ckpt_dir,exist_ok=True)
    os.makedirs(FIG_DIR,exist_ok=True)

    device=get_device()
    print("Device:",torch.cuda.get_device_name(0) if device=="cuda" else "cpu")

    x_train,_=load_eurosat_split("train")
    model=MAE().to(device)

    #correctness checks before any training
    sanity_check(model,x_train[:8].to(device))

    if os.path.exists(os.path.join(args.ckpt_dir,"mae_final.pt")):
        print("mae_final.pt already exists - delete it to retrain")
    else:
        pretrain(model,x_train,device,args.ckpt_dir,FIG_DIR,RESULTS_DIR,epochs=args.epochs)
