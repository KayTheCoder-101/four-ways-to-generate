"""Four ways to generate: one Gradio app with four tabs (MAE, VAE, Diffusion, RAG).

Run from the repo root:   python app/app.py            (add --share on Colab/Kaggle)
Weights are read from checkpoints/ if present, otherwise downloaded from Hugging Face.
"""
import os
import sys
import json
import argparse
import importlib.util
from functools import lru_cache

import numpy as np
import torch
import gradio as gr
from PIL import Image

HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(HERE)
for p in [os.path.join(ROOT,"starter"),os.path.join(ROOT,"part_d_rag")]:
    if p not in sys.path:
        sys.path.insert(0,p)

HF_REPO=os.environ.get("HF_REPO","KayTheCoderrrrrrr/four-ways-to-generate-weights")
LOCAL_DIRS=[os.path.join(ROOT,"checkpoints",d) for d in ("","mae","vae","ddpm","rag")]
ASSETS=os.path.join(HERE,"assets","uhat_examples.json")
NOTICE="For educational use only. Not medical advice."


#-----------helpers
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE=get_device()


def weight_path(filename,env=None):
    """Local checkpoints/ first, then the Hugging Face repo."""
    if env and os.environ.get(env) and os.path.exists(os.environ[env]):
        return os.environ[env]
    for d in LOCAL_DIRS:
        path=os.path.join(d,filename)
        if os.path.exists(path):
            return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_REPO,filename)


def load_module(alias,relpath):
    """Import a file by path, so part folders can all have their own model.py."""
    path=os.path.join(ROOT,relpath)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{relpath} not found")
    spec=importlib.util.spec_from_file_location(alias,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_pil(img,size=224):
    """img: [C,H,W] tensor in [0,1] -> upscaled PIL image (nearest, so pixels stay sharp)."""
    a=(img.detach().cpu().clamp(0,1)*255).byte()
    a=a.squeeze(0).numpy() if a.shape[0]==1 else a.permute(1,2,0).numpy()
    return Image.fromarray(a).resize((size,size),Image.NEAREST)


@lru_cache(None)
def uhat_assets():
    """One test image per character class + the 40 class names (made by make_uhat_assets.py)."""
    with open(ASSETS) as f:
        a=json.load(f)
    imgs=torch.tensor(a["images"],dtype=torch.float32).view(-1,1,28,28)/255.0
    return a["class_names"],imgs


def class_names():
    try:
        return uhat_assets()[0]
    except Exception:
        return [f"class {i}" for i in range(40)]


#-----------Tab 1: MAE (Part A)
@lru_cache(None)
def mae_bundle():
    m=load_module("part_a_model","part_a_mae/model.py")
    from eurosat import load_eurosat_split
    model=m.MAE().to(DEVICE)
    model.load_state_dict(torch.load(weight_path("mae_final.pt"),map_location=DEVICE))
    model.eval()
    clf=torch.nn.Linear(192,10).to(DEVICE)
    clf.load_state_dict(torch.load(weight_path("mae_classifier.pt"),map_location=DEVICE))
    clf.eval()
    x_te,y_te=load_eurosat_split("test")
    return model,clf,x_te,y_te


def mae_run(idx,seed):
    from eurosat import patchify,unpatchify,CLASS_NAMES
    model,clf,x_te,y_te=mae_bundle()
    idx=int(idx)
    torch.manual_seed(int(seed))   #the seed picks which 48 patches are hidden
    img=x_te[idx:idx+1].to(DEVICE)
    with torch.no_grad():
        _,pred,mask=model(img)
        patches=patchify(img)
        m=mask.unsqueeze(-1).bool()
        masked=unpatchify(torch.where(m,torch.full_like(patches,0.5),patches))
        recon=unpatchify(torch.where(m,pred,patches))
        probs=torch.softmax(clf(model.features(img)),dim=1)[0].cpu()
    label={CLASS_NAMES[i]:float(probs[i]) for i in range(10)}
    truth=f"**True class:** {CLASS_NAMES[int(y_te[idx])]}  ·  **Predicted:** {CLASS_NAMES[int(probs.argmax())]}"
    return to_pil(masked[0]),to_pil(recon[0]),to_pil(img[0]),label,truth


#-----------Tab 2: VAE (Part B)
@lru_cache(None)
def vae_model():
    m=load_module("part_b_model","part_b_vae/model.py")
    model=m.VAE().to(DEVICE)
    model.load_state_dict(torch.load(weight_path("vae_final.pt"),map_location=DEVICE))
    model.eval()
    return model


def vae_samples():
    model=vae_model()
    with torch.no_grad():
        z=torch.randn(16,16,device=DEVICE)               #z ~ N(0, I), latent dimension 16
        x=torch.sigmoid(model.decode(z))                  #decoder gives one logit per pixel
    return [to_pil(x[i],112) for i in range(16)]


def vae_interpolate(name_a,name_b,alpha):
    model=vae_model()
    names,imgs=uhat_assets()
    xa=imgs[names.index(name_a)][None].to(DEVICE)
    xb=imgs[names.index(name_b)][None].to(DEVICE)
    with torch.no_grad():
        mu_a,_=model.encode(xa)
        mu_b,_=model.encode(xb)
        z=(1-alpha)*mu_a+alpha*mu_b                       #point on the line between the two means
        x=torch.sigmoid(model.decode(z))
        steps=torch.linspace(0,1,10,device=DEVICE)[:,None]
        strip=torch.sigmoid(model.decode((1-steps)*mu_a+steps*mu_b))
    return to_pil(x[0],224),[to_pil(s,96) for s in strip]


#-----------Tab 3: Diffusion (Part C)
@lru_cache(None)
def ddpm_bundle():
    m=load_module("part_c_ddpm","part_c_diffusion/ddpm.py")
    unet=m.build_unet().to(DEVICE)
    unet.load_state_dict(torch.load(weight_path("ddpm_final.pt"),map_location=DEVICE))
    unet.eval()
    return m,unet


def ddpm_generate(name,seed,progress=gr.Progress()):
    m,unet=ddpm_bundle()
    torch.manual_seed(int(seed))
    label=class_names().index(name)
    progress(0,desc="Denoising 1000 steps...")
    with torch.no_grad():
        x=m.sample(unet,torch.full((8,),label,dtype=torch.long,device=DEVICE),DEVICE)   #[-1, 1]
    x=(x+1)/2
    return [to_pil(x[i],112) for i in range(8)]


#-----------Tab 4: RAG (Part D)
@lru_cache(None)
def rag_bundle():
    import embed
    import search as S
    import generate as G
    from transformers import AutoTokenizer,AutoModelForCausalLM
    idx=torch.load(weight_path("rag_index.pt",env="RAG_INDEX"),map_location="cpu")
    E=idx["emb"].to(DEVICE)
    tok_e,mod_e=embed.load_embedder(DEVICE)
    name="Qwen/Qwen2.5-1.5B-Instruct"
    dtype=torch.float16 if DEVICE in ("cuda","mps") else torch.float32
    tok_g=AutoTokenizer.from_pretrained(name)
    mod_g=AutoModelForCausalLM.from_pretrained(name,torch_dtype=dtype).to(DEVICE)
    mod_g.eval()
    return idx,E,tok_e,mod_e,tok_g,mod_g


def rag_run(question):
    import embed
    import search as S
    import generate as G
    from generation_eval import cited_numbers
    question=(question or "").strip()
    if not question:
        return "Type a question first.",""
    idx,E,tok_e,mod_e,tok_g,mod_g=rag_bundle()
    q=embed.embed([question],tok_e,mod_e)
    _,top=S.search(q,E,idx["abs_ids"],k_abs=10,k_chunk=5)
    top=top[0]
    passages=[idx["texts"][c] for c in top]
    ans=G.answer(question,passages,tok_g,mod_g)
    cited=set(cited_numbers(ans))

    answer_md=f"{ans}\n\n> ⚠️ **{NOTICE}**"
    lines=[]
    for i,c in enumerate(top):
        mark="✅ cited" if (i+1) in cited else "not cited"
        lines.append(f"**[{i+1}]** · abstract id `{idx['abs_ids'][c]}` · {mark}\n\n{passages[i]}")
    return answer_md,"\n\n---\n\n".join(lines)


def safe(fn,part):
    """Wrap a tab callback so a missing model shows a message instead of a crash."""
    def wrapped(*args,**kwargs):
        try:
            return fn(*args,**kwargs)
        except gr.Error:
            raise
        except Exception as e:
            raise gr.Error(f"{part} is not available yet - {type(e).__name__}: {e}")
    return wrapped


#-----------UI
def build():
    names=class_names()
    with gr.Blocks(title="Four ways to generate") as demo:
        gr.Markdown("# Four ways to generate\nMasked autoencoder · VAE · Diffusion · RAG — Generative AI, Assignment 04")

        with gr.Tab("MAE – satellite tiles"):
            gr.Markdown("Pick a EuroSAT test tile. The MAE hides 75% of its patches, reconstructs them, "
                        "and a linear classifier on the frozen encoder predicts the land-cover class.")
            with gr.Row():
                idx=gr.Slider(0,5399,value=0,step=1,label="Test tile index")
                seed=gr.Number(value=0,precision=0,label="Mask seed")
                btn=gr.Button("Run",variant="primary")
            with gr.Row():
                o_mask=gr.Image(label="Masked input (75% hidden)")
                o_rec=gr.Image(label="Reconstruction")
                o_org=gr.Image(label="Original")
            with gr.Row():
                o_lab=gr.Label(num_top_classes=3,label="Predicted land cover")
                o_true=gr.Markdown()
            btn.click(safe(mae_run,"The MAE tab"),[idx,seed],[o_mask,o_rec,o_org,o_lab,o_true])

        with gr.Tab("VAE – Urdu characters"):
            gr.Markdown("New characters decoded from random points z ~ N(0, I), and a walk through latent space between two characters.")
            b_new=gr.Button("New random characters",variant="primary")
            g_new=gr.Gallery(label="16 samples",columns=8,height="auto")
            b_new.click(safe(vae_samples,"The VAE tab"),None,g_new)
            with gr.Row():
                a=gr.Dropdown(names,value=names[0],label="From character")
                b=gr.Dropdown(names,value=names[min(5,len(names)-1)],label="To character")
                alpha=gr.Slider(0,1,value=0.5,step=0.05,label="Interpolation")
            with gr.Row():
                i_one=gr.Image(label="Decoded at this point")
                i_strip=gr.Gallery(label="10 equally spaced steps",columns=10,height="auto")
            for comp in (a,b,alpha):
                comp.change(safe(vae_interpolate,"The VAE tab"),[a,b,alpha],[i_one,i_strip])

        with gr.Tab("Diffusion – write a character"):
            gr.Markdown("Pick a character; the DDPM starts from pure noise and denoises for 1000 steps, guided by the class label.")
            with gr.Row():
                ch=gr.Dropdown(names,value=names[0],label="Character")
                d_seed=gr.Number(value=0,precision=0,label="Seed")
                d_btn=gr.Button("Generate 8 samples",variant="primary")
            d_out=gr.Gallery(label="Samples",columns=8,height="auto")
            d_btn.click(safe(ddpm_generate,"The diffusion tab"),[ch,d_seed],d_out)

        with gr.Tab("RAG – biomedical questions"):
            gr.Markdown(f"Ask a biomedical research question. The answer is generated only from the 5 retrieved PubMed passages.\n\n**{NOTICE}**")
            q=gr.Textbox(label="Question",placeholder="e.g. Is capmatinib effective for glioblastoma?")
            r_btn=gr.Button("Answer",variant="primary")
            r_ans=gr.Markdown(label="Answer")
            with gr.Accordion("Retrieved passages",open=True):
                r_pass=gr.Markdown()
            r_btn.click(safe(rag_run,"The RAG tab"),q,[r_ans,r_pass])
            q.submit(safe(rag_run,"The RAG tab"),q,[r_ans,r_pass])
    return demo


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--share",action="store_true",help="public link (needed on Colab/Kaggle)")
    parser.add_argument("--port",type=int,default=7860)
    args=parser.parse_args()
    print("Device:",DEVICE,"| weights from:",HF_REPO)
    build().queue().launch(share=args.share,server_port=args.port)
