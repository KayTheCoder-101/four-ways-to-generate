# Generative AI – Assignment 04
**Masked Autoencoders, VAEs, Diffusion and RAG — four generative models, one app**

Group members:
- Ume Kulsoom — Parts A (MAE) and D (RAG)
- Mobeen Rukhsar — Parts B (VAE) and C (Diffusion)

Links: [Medium blog](https://medium.com/@f230013/four-generative-models-built-mostly-by-hand-52ee02b691e0) · [LinkedIn post](ADD-LINKEDIN-POST-URL) · [Trained weights](https://huggingface.co/KayTheCoderrrrrrr/four-ways-to-generate-weights)

---

## Repository layout
| Folder | Contents |
|---|---|
| `starter/` | the given starter files, unchanged |
| `part_a_mae/` | model, pretraining, reconstructions, few-label classifier |
| `part_b_vae/` | model, training, reconstructions, samples, interpolation |
| `part_c_diffusion/` | schedule, training, sampling |
| `part_d_rag/` | chunking, index, search, retrieval metrics, generation, evaluation |
| `results/` | figures, tables, `rag_answers.jsonl`, `human_eval.csv`, `analysis.md`, `samples.md` |
| `app/` | front end (4 tabs) |

## Setup
```bash
pip install -r requirements.txt
```
EuroSAT and BioASQ download automatically. For UHaT, download `data.zip` from
[Zenodo](https://zenodo.org/records/3670611) and unzip it into `starter/data/uhat/`.

## Task 0 – Starter check
```bash
cd starter
python check_starter.py
```
Output:
```
EuroSAT train (16200, 3, 64, 64) test (5400, 3, 64, 64)
few-label subset: 160 images; patches per image: (8, 64, 192)
UHaT train (28328, 1, 28, 28) test (4747, 1, 28, 28) classes: 40
BioASQ corpus: 40181 abstracts; test questions: 707
example: Is capmatinib effective for glioblastoma? | relevant ids: [31776899]
```

![EuroSAT examples](results/eurosat_examples.png)
![UHaT examples](results/uhat_examples.png)

## How to run each part
- **Part A – MAE** (GPU; ~11 min on a T4):
  ```bash
  python part_a_mae/model.py                        # shape + correctness checks
  python part_a_mae/pretrain.py --ckpt_dir CKPT     # A.1, 100 epochs, resumable
  python part_a_mae/reconstruct.py --ckpt_dir CKPT  # A.2
  python part_a_mae/classifier.py --ckpt_dir CKPT   # A.3 + A.4
  ```
  `CKPT` is any folder (default `checkpoints/mae`). To skip training, download `mae_final.pt` and `mae_classifier.pt` from the weights link into it.
- **Part B – VAE** (GPU; ~30 s on a T4):
  ```bash
  cd part_b_vae
  python check_vae.py                                  # correctness checks
  python train.py --data-root ../starter/data/uhat     # B.1–B.3, 30 epochs -> vae.pt + loss curves
  python figures.py                                    # B.4 reconstructions, 64 samples, interpolation
  ```
- **Part C – Diffusion** (GPU; ~35 min on a T4):
  ```bash
  cd part_c_diffusion
  python check_diffusion.py      # schedule checks + forward-noising figure (C.1, C.2)
  python train.py                # C.3, 40 epochs -> ddpm.pt + loss curve
  python figures.py              # C.4, C.5: 40 × 8 sample grid, denoising sequence, blind-test sheet
  ```
- **Part D – RAG** (GPU; ~10 min on a T4 for everything):
  ```bash
  export RAG_INDEX=/path/to/index.pt                # optional, default part_d_rag/index.pt
  python part_d_rag/chunking.py         # D.1  chunks + check
  python part_d_rag/embed.py            # D.2  MiniLM + mean pooling -> index
  python part_d_rag/search.py           # D.3  top-20 abstracts, top-5 chunks per question
  python part_d_rag/retrieval_eval.py   # D.4  Hit/Recall/MRR + Recall@k plot + failure cases
  python part_d_rag/generate.py         # D.5  Qwen2.5-1.5B answers for the first 100 questions
  python part_d_rag/generation_eval.py  # D.6  ROUGE-L, citation precision, "I don't know" rate
  ```
  To skip `embed.py`, download `rag_index.pt` from the weights link and point `RAG_INDEX` at it.
- **App** (Gradio, 4 tabs; runs on CUDA, Apple MPS or CPU):
  ```bash
  python app/make_uhat_assets.py   # once, needs UHaT in starter/data/uhat: one example per character
  python app/app.py                # open http://127.0.0.1:7860   (add --share on Colab/Kaggle)
  ```
  Weights are taken from `checkpoints/` (or `part_b_vae/vae.pt`, `part_c_diffusion/ddpm.pt`) if present,
  otherwise downloaded from the Hugging Face repo. `RAG_INDEX=/path/to/index.pt` points the RAG tab at a local index.

## Trained weights
All weights: https://huggingface.co/KayTheCoderrrrrrr/four-ways-to-generate-weights

| File | Part | Place in |
|---|---|---|
| `mae_final.pt` | A – pretrained MAE (encoder + decoder) | `checkpoints/mae/` |
| `mae_classifier.pt` | A – few-label linear classifier | `checkpoints/mae/` |
| `vae.pt` | B – VAE (`{"model", "class_names"}`) | `part_b_vae/` |
| `ddpm.pt` | C – DDPM U-Net (`{"model", "class_names", "epoch"}`) | `part_c_diffusion/` |
| `rag_index.pt` | D – chunk embeddings (79,332 × 384), chunk texts and abstract ids | path in `RAG_INDEX` |

## Correctness checks
- **MAE:** exactly 48 of 64 patches masked per image; loss only on masked patches (predictions at visible positions replaced with noise, loss unchanged)
  ```
  CHECK masked patches per image: [48.0] (expected [48.0])
  CHECK loss only on masked patches: 0.517114 == 0.517114
  ```
- **VAE:** KL ≥ 0, and KL = 0 when µ = 0, log σ² = 0
  ```
  KL min over 1000 random posteriors: 13.632020950317383
  KL at mu=0, logvar=0: [-0.0, -0.0, -0.0, -0.0]
  loss check: total=554.10 recon=554.08 kl=0.02
  ALL VAE CHECKS PASSED
  ```
- **DDPM:** ᾱ_T ≈ 4×10⁻⁵, so x_T is almost pure noise (see the t = 999 panel of the forward-noising figure)
  ```
  betas: 0.0001 -> 0.02 (len 1000)
  alpha_bar at t=0: 0.999900
  alpha_bar_T (t=999): 4.036e-05   (expected about 4e-5)
  ```
- **RAG:** every chunk ≤ 150 words (consecutive chunks share 30); every embedding has norm 1; a perfect ranking gives Hit@k = Recall@k = MRR@10 = 1
  ```
  CHECK longest chunk: 150 words (<= 150); overlap = 30 words
  CHECK embedding norms: min 1.000000, max 1.000000 (all = 1)
  CHECK perfect ranking: {'Hit@1': 1.0, 'Recall@1': 1.0, 'Hit@5': 1.0, 'Recall@5': 1.0, 'Hit@10': 1.0, 'Recall@10': 1.0, 'MRR@10': 1.0}
  ```

## Results
### Table 1 – Data
| Item | Value |
|---|---|
| EuroSAT train / test images; labelled images used | 16,200 / 5,400; 160 |
| UHaT character classes; train / test images | 40; 28,328 / 4,747 |
| BioASQ abstracts; chunks; test questions | 40,181; 79,332; 707 |
| Mean number of relevant abstracts per test question | 8.66 |

### Table 2 – Training
| Model | Parameters | Final training loss | Time / GPU |
|---|---|---|---|
| MAE (encoder + decoder) | 3,962,752 | 0.0062 (masked MSE) | 10.7 min / Tesla T4 |
| VAE | 220,385 | 129.96 (negative ELBO, nats) | 25 s / Tesla T4 |
| DDPM U-Net | 4,028,353 | 0.0230 (L_simple) | 34.9 min / Tesla T4 |

### Table 3 – Image results
| Item | Value |
|---|---|
| MAE few-label classifier: test accuracy | 53.4% (chance 10%) |
| VAE: test negative ELBO = reconstruction + KL (nats) | 130.38 = 105.96 + 24.42 |
| Diffusion: blind test, member 1 / member 2 correct (of 40) | pending – filled by `results/merge_human_eval.py` |
| Diffusion: images where both members agreed (of 40) | pending – filled by `results/merge_human_eval.py` |

### Table 4 – Retrieval
| Hit@1 | Hit@5 | Hit@10 | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|---|
| 0.622 | 0.816 | 0.861 | 0.622 | 0.586 | 0.602 | 0.706 |

98 of 707 test questions (13.9%) have no relevant abstract in the top 10 (`results/rag_failure_cases.json`).

### Table 5 – Generation (100 questions)
| Item | Value |
|---|---|
| ROUGE-L | 0.233 |
| Citation precision | 0.586 (51 / 87 citations) |
| "I don't know" rate | 0.16 |
| Human: supported, member 1 / member 2 (of 20) | pending – filled by `results/merge_human_eval.py` |
| Human: correct, member 1 / member 2 (of 20) | pending – filled by `results/merge_human_eval.py` |

## Part A figures
![MAE pretraining loss](results/figures/mae_pretrain_loss.png)
![MAE reconstructions: masked input / reconstruction / original](results/figures/mae_reconstructions.png)
![MAE confusion matrix](results/figures/mae_confusion_matrix.png)

## Part B figures
![VAE loss terms](results/vae_loss_terms.png)
![VAE reconstructions](results/vae_reconstructions.png)
![VAE samples](results/vae_samples.png)
![VAE interpolation](results/vae_interpolation.png)

## Part C figures
![Forward noising](results/ddpm_forward.png)
![DDPM loss](results/ddpm_loss.png)
![40 × 8 samples](results/ddpm_samples_grid.png)
![Denoising](results/ddpm_denoising.png)
![Blind test: 40 shuffled samples, one per character](results/ddpm_blind_test.png)

## Part D figures
![RAG Recall@k](results/figures/rag_recall_at_k.png)

## Qualitative samples, analysis and human evaluation
- [`results/samples.md`](results/samples.md): five RAG questions with answer, cited passages and reference, including a retrieval failure (F105-P), an "I don't know" and a hallucination (capmatinib)
- [`results/analysis.md`](results/analysis.md): answers to E.1–E.7
- [`results/human_eval.csv`](results/human_eval.csv): diffusion blind test and RAG supported/correct ratings by both members, merged with `python results/merge_human_eval.py`

## App screenshots
| MAE | VAE |
|---|---|
| ![MAE tab](results/screenshots/app_mae.png) | ![VAE tab](results/screenshots/app_vae.png) |
| **Diffusion** | **RAG** |
| ![Diffusion tab](results/screenshots/app_diffusion.png) | ![RAG tab](results/screenshots/app_rag.png) |

> Part D answers are for educational use only. Not medical advice.