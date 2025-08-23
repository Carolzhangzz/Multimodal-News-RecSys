# scripts/eval_content_adapter.py
# -*- coding: utf-8 -*-
import argparse, os, json, time, csv, sys
import numpy as np, pandas as pd, torch
from tqdm import tqdm

# 允许直接运行
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train.train_content_adapter import ContentAdapter, build_user_profiles

def load_graph(graph_dir):
    with open(os.path.join(graph_dir, "meta.json")) as f:
        meta = json.load(f)
    train = pd.read_parquet(os.path.join(graph_dir, "train_edges.parquet"))
    valid = pd.read_parquet(os.path.join(graph_dir, "valid_pairs.parquet"))
    test  = pd.read_parquet(os.path.join(graph_dir, "test_pairs.parquet"))
    feats = np.load(os.path.join(graph_dir, "item_features_aligned.npz"))
    return meta, train, valid, test, feats

def build_user_pos_set(n_users, train_df):
    pos = [set() for _ in range(n_users)]
    for u,i in zip(train_df["u"].to_numpy(), train_df["i"].to_numpy()):
        pos[int(u)].add(int(i))
    return pos

@torch.no_grad()
def evaluate(U, I, pairs_df, user_pos, n_items, K=10, n_neg=99, device="cpu"):
    HR,NDCG=[],[]
    for u,i_pos in tqdm(pairs_df[["u","i"]].to_numpy(), desc="Eval(adapter)", leave=False):
        negs=set()
        while len(negs)<n_neg:
            j=np.random.randint(0,n_items)
            if (j not in user_pos[u]) and (j!=i_pos): negs.add(j)
        cand=np.array([i_pos]+list(negs),dtype=np.int64)
        u_idx=torch.tensor([u]*len(cand),dtype=torch.long,device=device)
        i_idx=torch.tensor(cand,dtype=torch.long,device=device)
        scores=(U[u_idx]*I[i_idx]).sum(-1).cpu().numpy()
        order=np.argsort(-scores); rank=int(np.where(order==0)[0][0])
        hit=1 if rank<K else 0
        HR.append(hit); NDCG.append(1/np.log2(rank+2) if hit else 0.0)
    return float(np.mean(HR)), float(np.mean(NDCG))

def main(args):
    device=torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    meta, train_df, valid_df, test_df, feats = load_graph(args.graph)
    n_users, n_items = meta["n_users"], meta["n_items"]

    U_raw_np, I_raw_np = build_user_profiles(n_users, train_df, feats["txt"], feats["price"])
    U_raw = torch.tensor(U_raw_np, dtype=torch.float32, device=device)
    I_raw = torch.tensor(I_raw_np, dtype=torch.float32, device=device)

    ckpt = torch.load(args.ckpt, map_location=device)
    d = ckpt.get("d", args.d)
    model = ContentAdapter(in_dim=int(U_raw.shape[1]), d=int(d)).to(device)
    model.load_state_dict(ckpt["state"])
    with torch.no_grad():
        _, U, I = model(U_raw, I_raw,
                        torch.tensor([0],dtype=torch.long,device=device),
                        torch.tensor([0],dtype=torch.long,device=device))

    user_pos = build_user_pos_set(n_users, train_df)

    u_cnt = train_df["u"].value_counts().reindex(range(n_users), fill_value=0)
    i_cnt = train_df["i"].value_counts().reindex(range(n_items), fill_value=0)
    cold_users = set(u_cnt[u_cnt==0].index.astype(int)) if args.extreme_cold_user else set(u_cnt[u_cnt<=args.cold_user_th].index.astype(int))
    cold_items = set(i_cnt[i_cnt==0].index.astype(int)) if args.extreme_cold_item else set(i_cnt[i_cnt<=args.cold_item_th].index.astype(int))

    tasks = {
        "valid_all": valid_df,
        "test_all":  test_df,
        "valid_cold_user": valid_df[valid_df["u"].isin(cold_users)],
        "test_cold_user":  test_df[test_df["u"].isin(cold_users)],
        "valid_cold_item": valid_df[valid_df["i"].isin(cold_items)],
        "test_cold_item":  test_df[test_df["i"].isin(cold_items)],
    }

    rows=[]
    for name, df in tasks.items():
        if len(df)==0: print(f"{name}: 0 samples, skip"); continue
        hr, nd = evaluate(U, I, df, user_pos, n_items, K=args.K, n_neg=args.n_neg, device=device)
        print(f"{name} (ADAPTER): HR@{args.K}={hr:.4f} NDCG@{args.K}={nd:.4f} (n={len(df)})")
        rows.append([time.strftime("%Y-%m-%d %H:%M:%S"), name, len(df), hr, nd,
                     args.cold_user_th, args.cold_item_th, int(args.extreme_cold_user), int(args.extreme_cold_item),
                     args.K, args.n_neg])

    out_csv = os.path.join(os.path.dirname(args.ckpt), "adapter_eval_log.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    newfile = not os.path.exists(out_csv)
    with open(out_csv,"a",newline="") as f:
        w=csv.writer(f)
        if newfile:
            w.writerow(["time","split","n","HR","NDCG","cold_user_th","cold_item_th","extreme_user","extreme_item","K","n_neg"])
        w.writerows(rows)
    print("Logged to:", out_csv)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--graph", default="data/graph")
    ap.add_argument("--ckpt",  default="runs/content_adapter/best.pt")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--n_neg", type=int, default=99)
    ap.add_argument("--cold_user_th", type=int, default=5)
    ap.add_argument("--cold_item_th", type=int, default=10)
    ap.add_argument("--extreme_cold_user", action="store_true")
    ap.add_argument("--extreme_cold_item", action="store_true")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--cpu", action="store_true")
    args=ap.parse_args()
    main(args)
