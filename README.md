# Generative AI – Assignment 04
**Masked Autoencoders, VAEs, Diffusion and RAG — four generative models, one app**

Group members:
- Member 1 — Parts A (MAE) and D (RAG)
- Member 2 — Parts B (VAE) and C (Diffusion)

Links: [Medium blog](TODO) · [LinkedIn post](TODO) · [Trained weights](TODO)

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
Output: TODO (paste every printed shape and count)

![EuroSAT examples](results/eurosat_examples.png)
![UHaT examples](results/uhat_examples.png)

## How to run each part
- **Part A – MAE:** TODO
- **Part B – VAE:** TODO
- **Part C – Diffusion:** TODO
- **Part D – RAG:** TODO
- **App:** TODO

## Trained weights
TODO – Hugging Face / Google Drive links and where to place them

## Correctness checks
- **MAE:** exactly 48 of 64 patches masked per image; loss only on masked patches — TODO output
- **VAE:** KL ≥ 0, and KL = 0 when µ = 0, log σ² = 0 — TODO output
- **DDPM:** ᾱ_T ≈ 4×10⁻⁵; x at t = 999 looks like noise — TODO output
- **RAG:** every chunk ≤ 150 words; every embedding has norm 1; perfect ranking gives Hit@k = Recall@k = MRR@10 = 1 — TODO output

## Results
### Table 1 – Data
| | |
|---|---|
| EuroSAT train / test images; labelled images used | |
| UHaT character classes; train / test images | |
| BioASQ abstracts; chunks; test questions | |
| Mean number of relevant abstracts per test question | |

### Table 2 – Training
| Model | Parameters | Final training loss | Time / GPU |
|---|---|---|---|
| MAE (encoder + decoder) | | | |
| VAE | | | |
| DDPM U-Net | | | |

### Table 3 – Image results
| | |
|---|---|
| MAE few-label classifier: test accuracy | |
| VAE: test negative ELBO = reconstruction + KL (nats) | |
| Diffusion: blind test, member 1 / member 2 correct (of 40) | |
| Diffusion: images where both members agreed (of 40) | |

### Table 4 – Retrieval
| Hit@1 | Hit@5 | Hit@10 | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|---|
| | | | | | | |

### Table 5 – Generation (100 questions)
| | |
|---|---|
| ROUGE-L | |
| Citation precision | |
| "I don't know" rate | |
| Human: supported, member 1 / member 2 (of 20) | |
| Human: correct, member 1 / member 2 (of 20) | |

## App screenshots
TODO – MAE tab, VAE tab, Diffusion tab, RAG tab

> Part D answers are for educational use only. Not medical advice.
