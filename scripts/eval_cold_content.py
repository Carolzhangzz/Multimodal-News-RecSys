# scripts/eval_cold_content.py
# -*- coding: utf-8 -*-
import argparse, os, json, numpy as np, pandas as pd, torch
from tqdm import tqdm

def load_all(graph_dir):
    with open(os.path.join(graph_dir, "meta.json")) as f:
        meta = json.load(f)
    train = pd.read_parquet(os.path.join(graph_dir, "train_edges.parquet"))
    valid = pd.read_parquet(os.path.join(graph_dir, "valid_pairs.parquet"))
    test  = pd.read_parquet(os.path.join(graph_dir, "test_pairs.parquet"))
    feats = np.load(os.path.join(graph_dir, "item_features_aligned.npz"))
    return meta, train, valid, test, feats

def build_user_profiles(n_users, train_df, item_txt):
    # item_txt: (n_items, 384)
    prof = np.zeros((n_users, item_txt.shape[1]), dtype=np.float32)
    cnt  = np.zeros(n_users, dtype=np.int32)
    for u, i in zip(train_df["u"].to_numpy(), train_df["i"].to_numpy()):
        prof[u] += item_txt[i]
        cnt[u]  += 1
    cnt = np.maximum(cnt, 1)
    prof = prof / cnt[:, None]
    # 归一化（做 cosine）
    prof = prof / (np.linalg.norm(prof, axis=1, keepdims=True) + 1e-12)
    items = item_txt / (np.linalg.norm(item_txt, axis=1, keepdims=True) + 1e-12)
    return prof.astype(np.float32), items.astype(np.float32)

def sample_eval(user_prof, item_txt, pairs_df, user_pos_set, n_items, K=10, n_neg=99):
    HR, NDCG = [], []
    for u, i_pos in tqdm(pairs_df[["u","i"]].to_numpy(), desc="Eval(content)", leave=False):
        negs = set()
        while len(negs) < n_neg:
            j = np.random.randint(0, n_items)
            if (j not in user_pos_set[u]) and (j != i_pos):
                negs.add(j)
        cand = np.array([i_pos] + list(negs), dtype=np.int64)
        scores = (item_txt[cand] @ user_prof[u])   # 余弦相似度
        order  = np.argsort(-scores)
        rank   = int(np.where(order == 0)[0][0])
        hit = 1 if rank < K else 0
        HR.append(hit); NDCG.append(1/np.log2(rank+2) if hit else 0.0)
    return float(np.mean(HR)), float(np.mean(NDCG))

def build_user_pos_set(n_users, train_df):
    pos = [set() for _ in range(n_users)]
    for u,i in zip(train_df["u"].to_numpy(), train_df["i"].to_numpy()):
        pos[int(u)].add(int(i))
    return pos

def main(args):
    meta, train_df, valid_df, test_df, feats = load_all(args.graph)
    n_users, n_items = meta["n_users"], meta["n_items"]
    user_prof, item_txt = build_user_profiles(n_users, train_df, feats["txt"])
    user_pos = build_user_pos_set(n_users, train_df)

    # 冷集（与你现有脚本一致）
    u_cnt = train_df["u"].value_counts().reindex(range(n_users), fill_value=0)
    i_cnt = train_df["i"].value_counts().reindex(range(n_items), fill_value=0)
    if args.extreme_cold_item:
        cold_items = set(i_cnt[i_cnt == 0].index.astype(int).tolist())
    else:
        cold_items = set(i_cnt[i_cnt <= args.cold_item_th].index.astype(int).tolist())
    if args.extreme_cold_user:
        cold_users = set(u_cnt[u_cnt == 0].index.astype(int).tolist())
    else:
        cold_users = set(u_cnt[u_cnt <= args.cold_user_th].index.astype(int).tolist())

    tasks = {
        "valid_all": valid_df,
        "test_all":  test_df,
        "valid_cold_user": valid_df[valid_df["u"].isin(cold_users)],
        "test_cold_user":  test_df[test_df["u"].isin(cold_users)],
        "valid_cold_item": valid_df[valid_df["i"].isin(cold_items)],
        "test_cold_item":  test_df[test_df["i"].isin(cold_items)],
    }
    for name, df in tasks.items():
        if len(df)==0: print(name, "0 samples, skip"); continue
        hr, nd = sample_eval(user_prof, item_txt, df, user_pos, n_items, K=args.K, n_neg=args.n_neg)
        print(f"{name} (CONTENT): HR@{args.K}={hr:.4f} NDCG@{args.K}={nd:.4f} (n={len(df)})")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="data/graph")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--n_neg", type=int, default=99)
    ap.add_argument("--cold_user_th", type=int, default=5)
    ap.add_argument("--cold_item_th", type=int, default=10)
    ap.add_argument("--extreme_cold_user", action="store_true")
    ap.add_argument("--extreme_cold_item", action="store_true")
    args = ap.parse_args()
    main(args)