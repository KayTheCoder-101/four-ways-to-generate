"""Part A - A.3 and A.4: linear classifier on the frozen encoder, trained on
160 labelled images (16 per class); test accuracy and confusion matrix."""
import os
import json
import argparse
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import load_trained_mae,get_device,RESULTS_DIR,FIG_DIR,DEFAULT_CKPT_DIR
from eurosat import load_eurosat_split,few_label_indices,CLASS_NAMES


#-----------A.3: few-label classifier
def few_label_classifier(model,x_train,y_train,device):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    idx=few_label_indices(y_train)
    labels=y_train[idx].to(device)

    #features never change, so compute them once and reuse
    with torch.no_grad():
        feats=model.features(x_train[idx].to(device))
    print("Few-label features:",tuple(feats.shape))

    clf=nn.Linear(192,10).to(device)
    optimizer=torch.optim.AdamW(clf.parameters(),lr=1e-3)
    criterion=nn.CrossEntropyLoss()

    for epoch in range(1,201):
        loss=criterion(clf(feats),labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch%20==0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    return clf


#-----------A.4: evaluation
def evaluate(model,clf,x_test,y_test,device,fig_dir,results_dir):
    class_names=CLASS_NAMES

    all_feats=[]
    with torch.no_grad():
        for start in range(0,len(x_test),512):
            batch=x_test[start:start+512].to(device)
            all_feats.append(model.features(batch))
        all_feats=torch.cat(all_feats,dim=0)

        preds=clf(all_feats).argmax(dim=1)

    y_true=y_test.to(device)
    accuracy=(preds==y_true).float().mean().item()
    print(f"Test Accuracy: {accuracy:.4f}")

    #confusion matrix by hand (rows = true class, columns = predicted class)
    cm=torch.bincount(y_true*10+preds,minlength=100).reshape(10,10).cpu()

    plt.figure(figsize=(8,7))
    plt.imshow(cm.numpy(),cmap="Blues")
    plt.colorbar()
    plt.xticks(range(10),class_names,rotation=45,ha="right")
    plt.yticks(range(10),class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"MAE few-label classifier (test acc {accuracy:.3f})")
    for i in range(10):
        for j in range(10):
            plt.text(j,i,int(cm[i,j]),ha="center",va="center",fontsize=7)
    plt.savefig(os.path.join(fig_dir,"mae_confusion_matrix.png"),bbox_inches="tight")
    plt.close()

    with open(os.path.join(results_dir,"mae_classifier_results.json"),"w") as f:
        json.dump({"accuracy":accuracy,"class_names":class_names,"confusion_matrix":cm.tolist()},f,indent=4)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir",default=DEFAULT_CKPT_DIR)
    args=parser.parse_args()

    torch.manual_seed(42)
    os.makedirs(FIG_DIR,exist_ok=True)
    device=get_device()

    x_train,y_train=load_eurosat_split("train")
    x_test,y_test=load_eurosat_split("test")
    model=load_trained_mae(args.ckpt_dir,device)

    clf=few_label_classifier(model,x_train,y_train,device)
    torch.save(clf.state_dict(),os.path.join(args.ckpt_dir,"mae_classifier.pt"))
    evaluate(model,clf,x_test,y_test,device,FIG_DIR,RESULTS_DIR)
