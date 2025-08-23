# train/train_mmgcn.py
# -*- coding: utf-8 -*-
import argparse, json, os, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from tqdm import tqdm
from models.mmgcn_light import MMGCN, SparseProp

SEED = 2025
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def load_graph(graph_dir):
    with open(os.path.join(graph_dir, "meta.json")) as f:
        meta = json.load(f)
    train = pd.read_parquet(os.path.join(graph_dir, "train_edges.parquet"))
    valid = pd.read_parquet(os.path.join(graph_dir, "valid_pairs.parquet"))
    test  = pd.read_parquet(os.path.join(graph_dir, "test_pairs.parquet"))
    feats = np.load(os.path.join(graph_dir, "item_features_aligned.npz"))
    n_users, n_items = meta["n_users"], meta["n_items"]
    # 训练边（同构图坐标：物品整体 +n_users）
    ui = torch.tensor(
        np.stack([train["u"].to_numpy(), train["i"].to_numpy() + n_users], axis=0),
        dtype=torch.long
    )
    return meta, train, valid, test, feats, ui

class BPRLoss(nn.Module):
    def forward(self, pos, neg):
        return -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()

def build_user_pos_set(n_users, train_df):
    pos = [set() for _ in range(n_users)]
    for u, i in zip(train_df["u"].to_numpy(), train_df["i"].to_numpy()):
        pos[int(u)].add(int(i))
    return pos

def sample_negatives(n_items, user_pos_set, u_batch):
    neg_items = []
    for u in u_batch:
        s = user_pos_set[u]
        j = np.random.randint(0, n_items)
        while j in s:
            j = np.random.randint(0, n_items)
        neg_items.append(j)
    return np.array(neg_items, dtype=np.int64)

def evaluate(model, U, I, pairs_df, user_pos_set, n_items, K=10, n_neg=99, device="cpu"):
    """采样评估：每个 (u, i_pos) + 99 负例，算 HR@K / NDCG@K"""
    model.eval()
    HR, NDCG = [], []
    for u, i_pos in tqdm(pairs_df[["u", "i"]].to_numpy(), desc="Eval", leave=False):
        # 负例去重
        negs = set()
        while len(negs) < n_neg:
            j = np.random.randint(0, n_items)
            if (j not in user_pos_set[u]) and (j != i_pos):
                negs.add(j)
        cand = np.array([i_pos] + list(negs), dtype=np.int64)

        u_idx = torch.tensor([u] * len(cand), dtype=torch.long, device=device)
        i_idx = torch.tensor(cand, dtype=torch.long, device=device)
        scores = model.score(U, I, u_idx, i_idx).detach().cpu().numpy()

        # 正样本在降序中的名次（0-based）
        order = np.argsort(-scores)
        rank = int(np.where(order == 0)[0][0])
        hit = 1 if rank < K else 0
        hr = hit
        ndcg = 1 / np.log2(rank + 2) if hit else 0.0
        HR.append(hr); NDCG.append(ndcg)
    return float(np.mean(HR)), float(np.mean(NDCG))

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.out, exist_ok=True)

    meta, train_df, valid_df, test_df, feats, ui_edges = load_graph(args.graph)
    n_users, n_items = meta["n_users"], meta["n_items"]

    # 稀疏传播器（基于 train 边）
    prop = SparseProp(ui_edges, n_users + n_items)

    # item 特征张量
    x_txt = torch.tensor(feats["txt"],   dtype=torch.float32, device=device)
    x_img = torch.tensor(feats["img"],   dtype=torch.float32, device=device)
    x_st  = torch.tensor(feats["price"], dtype=torch.float32, device=device)

    model = MMGCN(
        n_users, n_items, d=args.d, L=args.layers,
        txt_dim=x_txt.shape[1], img_dim=x_img.shape[1], struct_dim=x_st.shape[1]
    ).to(device)
    model.set_propagator(prop)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = BPRLoss()

    user_pos = build_user_pos_set(n_users, train_df)
    u_arr = train_df["u"].to_numpy(dtype=np.int64)
    i_arr = train_df["i"].to_numpy(dtype=np.int64)

    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        idx = np.random.permutation(len(u_arr))
        u_arr_s, i_arr_s = u_arr[idx], i_arr[idx]

        losses = []
        for s in range(0, len(u_arr_s), args.batch):
            ub = u_arr_s[s:s + args.batch]
            ib = i_arr_s[s:s + args.batch]
            jb = sample_negatives(n_items, user_pos, ub)

            # ⭐ 每个 batch 重新 encode，避免二次反传同一计算图
            U, I = model.encode(x_txt, x_img, x_st, device)

            u_t = torch.tensor(ub, dtype=torch.long, device=device)
            i_t = torch.tensor(ib, dtype=torch.long, device=device)
            j_t = torch.tensor(jb, dtype=torch.long, device=device)

            pos = model.score(U, I, u_t, i_t)
            neg = model.score(U, I, u_t, j_t)

            optim.zero_grad()
            loss = crit(pos, neg)
            loss.backward()
            optim.step()
            losses.append(loss.item())

        avg_loss = float(np.mean(losses)) if losses else 0.0

        # 验证（前向、无梯度）
        with torch.no_grad():
            U_eval, I_eval = model.encode(x_txt, x_img, x_st, device)
            hr, nd = evaluate(model, U_eval, I_eval, valid_df, user_pos, n_items, K=10, n_neg=99, device=device)
        print(f"Epoch {epoch:02d} | loss {avg_loss:.4f} | valid HR@10 {hr:.4f} NDCG@10 {nd:.4f}")

        if hr + nd > best_val:
            best_val = hr + nd
            torch.save({"state": model.state_dict(), "meta": meta}, os.path.join(args.out, "best.pt"))

    # 测试
    ckpt = torch.load(os.path.join(args.out, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["state"])
    with torch.no_grad():
        U_final, I_final = model.encode(x_txt, x_img, x_st, device)
        hr, nd = evaluate(model, U_final, I_final, test_df, user_pos, n_items, K=10, n_neg=99, device=device)
    print(f"TEST HR@10 {hr:.4f} NDCG@10 {nd:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="data/graph")
    ap.add_argument("--out", default="runs/mmgcn")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    main(args)
