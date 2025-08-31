# scripts/eval_cold.py
# -*- coding: utf-8 -*-
"""
Evaluate on all / cold-user / cold-item (可切 extreme cold)；CSV 日志。
Run:
  python -m scripts.eval_cold --graph data/graph_full \
    --ckpt ckpt_mmgcn_light_full/mmgcn_light.pt --K 20 --n_neg 2000
"""
import argparse, os, json, time, csv, sys, importlib.util
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = "/content/drive/MyDrive/MMREC"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 优先尝试常规 import
try:
    from models.mmgcn_light import MMGCN, SparseProp
except ModuleNotFoundError:
    # 动态兜底：直接从文件加载（适配你之前的文件名）
    for cand in [
        os.path.join(PROJECT_ROOT, "models", "mmgcn_light.py"),
        os.path.join(PROJECT_ROOT, "mmgcn_train.py"),
        os.path.join(PROJECT_ROOT, "04_train_mmgcn_light.py"),
    ]:
        if os.path.exists(cand):
            spec = importlib.util.spec_from_file_location("mmgcn_dyn", cand)
            mm = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mm)
            MMGCN, SparseProp = mm.MMGCN, mm.SparseProp
            break
    else:
        raise  # 还是找不到就让它报错

def load_graph(graph_dir):
    with open(os.path.join(graph_dir, "meta.json")) as f:
        meta = json.load(f)
    train = pd.read_parquet(os.path.join(graph_dir, "train_edges.parquet"))
    valid = pd.read_parquet(os.path.join(graph_dir, "valid_pairs.parquet"))
    test  = pd.read_parquet(os.path.join(graph_dir, "test_pairs.parquet"))
    feats = np.load(os.path.join(graph_dir, "item_features_aligned.npz"))
    n_users, n_items = meta["n_users"], meta["n_items"]
    ui = torch.tensor(
        np.stack([train["u"].to_numpy(), train["i"].to_numpy() + n_users], axis=0),
        dtype=torch.long
    )
    return meta, train, valid, test, feats, ui

def build_user_pos_set(n_users, train_df):
    pos = [set() for _ in range(n_users)]
    for u, i in zip(train_df["u"].to_numpy(), train_df["i"].to_numpy()):
        pos[int(u)].add(int(i))
    return pos

@torch.no_grad()
def evaluate(U, I, pairs_df, user_pos_set, n_items, K=20, n_neg=1000, device="cpu"):
    HR, NDCG = [], []
    rng = np.random.default_rng(2025)
    for u, i_pos in tqdm(pairs_df[["u","i"]].to_numpy(), desc="Eval", leave=False):
        negs = set()
        while len(negs) < n_neg:
            j = int(rng.integers(0, n_items))
            if (j not in user_pos_set[u]) and (j != i_pos):
                negs.add(j)
        cand = np.array([i_pos] + list(negs), dtype=np.int64)
        u_idx = torch.tensor([u]*len(cand), dtype=torch.long, device=device)
        i_idx = torch.tensor(cand,          dtype=torch.long, device=device)
        scores = (U[u_idx] * I[i_idx]).sum(-1)
        order = torch.argsort(scores, descending=True)
        rank = int((order == 0).nonzero(as_tuple=True)[0].item())
        hit = 1 if rank < K else 0
        HR.append(hit)
        NDCG.append(1/np.log2(rank+2) if hit else 0.0)
    return float(np.mean(HR)), float(np.mean(NDCG))

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    meta, train_df, valid_df, test_df, feats, ui_edges = load_graph(args.graph)
    n_users, n_items = meta["n_users"], meta["n_items"]

    # propagator（放同一 device）
    prop = SparseProp(ui_edges.to(device), n_users + n_items, device=device)

    # features
    x_txt = torch.tensor(feats["txt"],      dtype=torch.float32, device=device)
    x_img = torch.tensor(feats["img"],      dtype=torch.float32, device=device)
    x_st  = torch.tensor(feats["price"],    dtype=torch.float32, device=device)
    img_mask = torch.tensor(feats["img_mask"], dtype=torch.float32, device=device) \
               if "img_mask" in feats.files else None

    # model & weights（优先用 ckpt.meta 的维度）
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model", ckpt.get("state"))
    meta_ckpt = ckpt.get("meta", {})
    d = meta_ckpt.get("dim", args.d)
    L = meta_ckpt.get("layers", args.layers)

    model = MMGCN(n_users, n_items, d=d, L=L,
                  txt_dim=x_txt.shape[1], img_dim=x_img.shape[1], struct_dim=x_st.shape[1]).to(device)
    model.set_propagator(prop)
    model.load_state_dict(state, strict=False)
    model.eval()

    with torch.no_grad():
        U, I = model.encode(x_txt, x_img, x_st, device=device, item_img_mask=img_mask)

    # 正样本集合
    user_pos = build_user_pos_set(n_users, train_df)

    # 训练出现次数
    u_cnt = train_df["u"].value_counts().reindex(range(n_users), fill_value=0)
    i_cnt = train_df["i"].value_counts().reindex(range(n_items), fill_value=0)

    # 冷定义
    cold_users = set(u_cnt[u_cnt == 0].index.astype(int)) if args.extreme_cold_user \
                 else set(u_cnt[u_cnt <= args.cold_user_th].index.astype(int))
    cold_items = set(i_cnt[i_cnt == 0].index.astype(int)) if args.extreme_cold_item \
                 else set(i_cnt[i_cnt <= args.cold_item_th].index.astype(int))

    tasks = {
        "valid_all": valid_df,
        "test_all":  test_df,
        "valid_cold_user": valid_df[valid_df["u"].isin(cold_users)],
        "test_cold_user":  test_df[test_df["u"].isin(cold_users)],
        "valid_cold_item": valid_df[valid_df["i"].isin(cold_items)],
        "test_cold_item":  test_df[test_df["i"].isin(cold_items)],
    }

    rows = []
    for name, df in tasks.items():
        if len(df) == 0:
            print(f"{name}: 0 samples, skip")
            continue
        hr, nd = evaluate(U, I, df, user_pos, n_items,
                          K=args.K, n_neg=args.n_neg, device=device)
        print(f"{name}: HR@{args.K}={hr:.4f}  NDCG@{args.K}={nd:.4f}  (n={len(df)})")
        rows.append([time.strftime("%Y-%m-%d %H:%M:%S"), name, len(df), hr, nd,
                     args.cold_user_th, args.cold_item_th,
                     int(args.extreme_cold_user), int(args.extreme_cold_item),
                     args.K, args.n_neg])

    out_csv = os.path.join(os.path.dirname(args.ckpt), "cold_eval_log.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    newfile = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if newfile:
            w.writerow(["time","split","n","HR","NDCG",
                        "cold_user_th","cold_item_th","extreme_user","extreme_item",
                        "K","n_neg"])
        w.writerows(rows)
    print("Logged to:", out_csv)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="data/graph")
    ap.add_argument("--ckpt",  default="runs/mmgcn/best.pt")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--n_neg", type=int, default=1000)
    ap.add_argument("--cold_user_th", type=int, default=1)
    ap.add_argument("--cold_item_th", type=int, default=1)
    ap.add_argument("--extreme_cold_user", action="store_true")
    ap.add_argument("--extreme_cold_item", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    main(args)
