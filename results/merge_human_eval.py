"""Merge both human evaluations into results/human_eval.csv and print the numbers for Tables 3 and 5.

Lives in results/. Run from the repo root once BOTH members have filled their columns:
    python results/merge_human_eval.py

Inputs (all in results/ unless noted):
  human_eval_rag.csv          20 RAG answers; supported_<name> / correct_<name> filled with 1 or 0
  human_eval_diffusion.csv    40 blind-test images; member1_guess / member2_guess = character names
  part_c_diffusion/blind_key.csv   the answer key (image_id, true_label) written by figures.py.
                              If it is missing, the key is rebuilt from the same seeded shuffle.
"""
import os
import pandas as pd

RES=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(RES)
PEOPLE=["kulsoom","mobeen"]                 #member1 = kulsoom, member2 = mobeen in the diffusion sheet
YES={"1":1,"0":0,"1.0":1,"0.0":0,"y":1,"n":0,"yes":1,"no":0,"true":1,"false":0}

#the 40 UHaT classes in loader order (sorted), as the DDPM was trained
CLASS_NAMES=["alif","alif mad aa","ayn","baa","bari yaa","cheey","choti yaa","daal","dhaal","faa",
             "gaaf","ghain","haa1","haa2","haa3","hamza","jeem","kaaf","khaa","laam",
             "meem","noon","noonghunna","paa","qaaf","raa","rhraa","seen","seey","sheen",
             "swaad","taa","ttaa","twa","waw","zaaa","zaal","zhaa","zwaa","zwaad"]


def norm(x):
    return str(x).strip().lower() if pd.notna(x) else ""


def to01(series,col):
    vals=series.map(lambda v:YES.get(norm(v)))
    if vals.isna().any():
        bad=series[vals.isna()].tolist()
        raise ValueError(f"{col}: use 1/0 (or yes/no); found {bad}")
    return vals.astype(int)


def blind_key():
    path=os.path.join(ROOT,"part_c_diffusion","blind_key.csv")
    if os.path.exists(path):
        print("answer key: part_c_diffusion/blind_key.csv")
        return pd.read_csv(path)
    #same shuffle as part_c_diffusion/figures.py (seed 0 -> generator seed 3)
    import torch
    perm=torch.randperm(40,generator=torch.Generator().manual_seed(3)).tolist()
    print("answer key: rebuilt from the seeded shuffle (blind_key.csv not found)")
    return pd.DataFrame({"image_id":range(1,41),"true_label":[CLASS_NAMES[p] for p in perm]})


#---------- RAG: 20 answers, supported / correct per member
rag=pd.read_csv(os.path.join(RES,"human_eval_rag.csv"))
for p in PEOPLE:
    for k in ("supported","correct"):
        col=f"{k}_{p}"
        rag[col]=to01(rag[col],col)

print("\nTable 5 – human evaluation of 20 RAG answers")
for k in ("supported","correct"):
    counts=" / ".join(str(int(rag[f"{k}_{p}"].sum())) for p in PEOPLE)
    agree=(rag[f"{k}_{PEOPLE[0]}"]==rag[f"{k}_{PEOPLE[1]}"]).sum()
    print(f"  Human: {k}, member 1 / member 2 (of 20): {counts}    (raters agree on {agree}/20)")

#---------- Diffusion: blind recognisability test
d=pd.read_csv(os.path.join(RES,"human_eval_diffusion.csv"))
d=d.rename(columns={"member1_guess":f"guess_{PEOPLE[0]}","member2_guess":f"guess_{PEOPLE[1]}"})
for p in PEOPLE:
    empty=(d[f"guess_{p}"].map(norm)=="").sum()
    if empty:
        raise ValueError(f"diffusion sheet: {p} still has {empty} empty guesses")
d=d.merge(blind_key(),on="image_id")
for p in PEOPLE:
    d[f"right_{p}"]=(d[f"guess_{p}"].map(norm)==d["true_label"].map(norm)).astype(int)
both_agree=(d[f"guess_{PEOPLE[0]}"].map(norm)==d[f"guess_{PEOPLE[1]}"].map(norm)).sum()

print("\nTable 3 – diffusion blind test (40 images)")
print(f"  Diffusion: blind test, member 1 / member 2 correct (of 40): "
      f"{d[f'right_{PEOPLE[0]}'].sum()} / {d[f'right_{PEOPLE[1]}'].sum()}")
print(f"  Diffusion: images where both members agreed (of 40): {both_agree}")
missed=d[(d[f"right_{PEOPLE[0]}"]==0)&(d[f"right_{PEOPLE[1]}"]==0)]
if len(missed):
    print("  missed by both (for E.5):",", ".join(f"{t} (read as {g})" for t,g in zip(missed["true_label"],missed[f"guess_{PEOPLE[0]}"])))

#---------- one sheet: diffusion rows, then RAG rows
diff_rows=pd.DataFrame({"task":"diffusion_blind_test","item_id":d["image_id"],"true_label":d["true_label"]})
for p in PEOPLE:
    diff_rows[f"guess_{p}"]=d[f"guess_{p}"]
    diff_rows[f"right_{p}"]=d[f"right_{p}"]
rag_rows=pd.DataFrame({"task":"rag_answer","item_id":rag["qid"],"question":rag["question"],
                       "answer":rag["answer"],"reference":rag["reference"]})
for p in PEOPLE:
    for k in ("supported","correct"):
        rag_rows[f"{k}_{p}"]=rag[f"{k}_{p}"]
out=pd.concat([diff_rows,rag_rows],ignore_index=True)
for c in out.columns:
    if c.startswith(("right_","supported_","correct_")):
        out[c]=out[c].astype("Int64")   #keep 1/0, blank where a column does not apply
out.to_csv(os.path.join(RES,"human_eval.csv"),index=False)
print(f"\nWrote results/human_eval.csv ({len(diff_rows)} diffusion rows + {len(rag_rows)} RAG rows)")
