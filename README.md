# Multimodal RecSys: MMGCN (Light) + Content Baselines on Amazon Reviews’23 (Fashion)

> A compact, **reproducible** baseline for cold-start recommendation that combines a lightweight LightGCN-style model with **content-only** methods (text + price) and simple **score fusion**. Designed to run on CPU (no PyG) and scale up on GPU when available.

---

![arch_system_design](https://github.com/Carolzhangzz/Multimodal-News-RecSys/multimodal.jpg)

## 1) What this project is

- **Goal.** Study **cold-start** (especially new items) on Amazon Reviews’23 **Amazon_Fashion**.
- **Models.**
  - **MMGCN‑Light**: a small LightGCN‑style graph model; item side uses text (MiniLM) + price.
  - **Content baselines**: (i) cosine(user profile, item text), (ii) a tiny **content adapter** that learns to map profiles/items.
  - **Fusion**: linear (α) & Reciprocal Rank Fusion (RRF).
- **Why this matters.** On **extreme cold items (unseen in train)** content signals dominate; GCN collapses. We provide a clean, reproducible setup to quantify that gap and explore simple fusions.

---

## 2) TL;DR results (sampling eval: 1 positive + 99 negatives, K=10)

| Split                 | Method            |   HR@10    |  NDCG@10   |
| --------------------- | ----------------- | :--------: | :--------: |
| **test_all**          | MMGCN‑Light (GCN) |   ~0.130   |   ~0.089   |
| **test_all**          | Content (cosine)  | **~0.241** | **~0.154** |
| **test_all**          | Content Adapter   | **~0.250** |   ~0.138   |
| **extreme cold item** | MMGCN‑Light (GCN) |   ~0.009   |   ~0.005   |
| **extreme cold item** | Content (cosine)  | **~0.213** | **~0.124** |

> Takeaway: **Content dominates** in cold start. Linear fusion only helps when α→0（i.e., mostly content).

---

## 3) Repo layout (key files)

```
multimodal_recsys/
├─ scripts/
│  ├─ prepare_subset_v2.py      # sample/clean to 3 parquet tables
│  ├─ extract_features.py       # text (MiniLM), (optional image), price
│  ├─ 03_build_graph.py         # leave-one-out split + index mapping
│  ├─ eval_cold.py              # GCN cold-start eval
│  ├─ eval_cold_content.py      # content-only (cosine) eval
│  ├─ eval_fusion.py            # GCN + content (linear/RRF)
│  ├─ eval_fusion_adapter.py    # GCN + content-adapter fusion
│  └─ __init__.py
├─ train/
│  ├─ train_mmgcn.py            # train the LightGCN-style model
│  ├─ train_content_adapter.py  # train tiny content adapter
│  └─ __init__.py
├─ models/
│  ├─ mmgcn_light.py            # model + sparse propagator
│  └─ __init__.py
└─ data/ (created locally; DO NOT commit large files)
```

---

## 4) Quick reproduce (CPU friendly)

> **Deps**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pandas pyarrow tqdm numpy sentence-transformers transformers open_clip_torch
```

> **Data** (download from Amazon Reviews’23 → **Amazon_Fashion**)

```
data/raw/
  Amazon_Fashion.jsonl.gz
  meta_Amazon_Fashion.jsonl.gz
```

```bash
gunzip -k data/raw/Amazon_Fashion.jsonl.gz
gunzip -k data/raw/meta_Amazon_Fashion.jsonl.gz
```

> **Prepare subset (example: 50k users)**

```bash
python scripts/prepare_subset_v2.py \
  --reviews data/raw/Amazon_Fashion.jsonl \
  --meta    data/raw/meta_Amazon_Fashion.jsonl \
  --out     data/processed \
  --sample_users 50000
```

> **Extract features (text + price; no images)**

```bash
python scripts/extract_features.py \
  --items data/processed/items.parquet \
  --interactions data/processed/interactions.parquet \
  --out data/features \
  --no-image --only-interacted --txt-bs 128
```

> **Build graph + LOO split**

```bash
python scripts/03_build_graph.py \
  --interactions data/processed/interactions.parquet \
  --items        data/processed/items.parquet \
  --item_features data/features/item_features.parquet \
  --out          data/graph
```

> **Train GCN baseline**

```bash
touch models/__init__.py train/__init__.py scripts/__init__.py
python -m train.train_mmgcn \
  --graph data/graph --out runs/mmgcn \
  --epochs 3 --batch 1024 --layers 2 --d 64 --cpu
```

> **Evaluate (GCN & cold start)**

```bash
python -m scripts.eval_cold --graph data/graph --ckpt runs/mmgcn/best.pt --cpu
python -m scripts.eval_cold --graph data/graph --ckpt runs/mmgcn/best.pt --extreme_cold_item --cpu
```

> **Content-only baseline**

```bash
python -m scripts.eval_cold_content --graph data/graph --extreme_cold_item
```

> **Fusion (linear + z-normalization)**

```bash
python -m scripts.eval_fusion \
  --graph data/graph --ckpt runs/mmgcn/best.pt \
  --alphas 0.0,0.25,0.5,0.75,1.0 --norm z --fusion linear --cpu
```

> **(Optional) Content adapter: train + fuse**

```bash
python -m train.train_content_adapter \
  --graph data/graph --out runs/content_adapter \
  --epochs 5 --batch 4096 --d 128 --cpu

python -m scripts.eval_fusion_adapter \
  --graph data/graph \
  --ckpt runs/mmgcn/best.pt \
  --adapter_ckpt runs/content_adapter/best.pt \
  --alphas 0.0,0.1,0.2,0.3,0.4,0.5 --norm z --fusion linear --cpu
```

**Notes**

- Add `--cpu` for CPU-only; remove it to use GPU.
- Increase `--n_neg` (e.g., 999) for more stable top‑K metrics.

---

## 5) What to report

- Dataset: Amazon Reviews’23 → Amazon_Fashion; sampling eval (1+99), K=10, seed=2025.
- Methods: **GCN baseline**, **Content-only**, **Content Adapter**, **Fusion (linear/RRF)**.
- Slices: **All**, **Cold‑User(≤5)**, **Cold‑Item(≤10 & =0)**.
- Key insight: **Content wins** in cold start; GCN adds little unless items are warm.

---

## 6) Git hygiene

Add large paths to `.gitignore` to avoid GitHub 100MB limit:

```
.venv/
__pycache__/
*.pyc
data/
runs/
*.pt
*.npz
```

If you accidentally committed large files, rewrite history (e.g., `git filter-repo`) or use Git LFS.

---

## 7) Citation

Hou, Yupeng; Li, Jiacheng; He, Zhankui; Yan, An; Chen, Xiusi; McAuley, Julian.  
*Bridging Language and Items for Retrieval and Recommendation*. arXiv:2403.03952, 2024.

---

## 8) License

Research/educational use only.
