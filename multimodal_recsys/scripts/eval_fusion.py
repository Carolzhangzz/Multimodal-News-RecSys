# scripts/eval_fusion.py
# -*- coding: utf-8 -*-
"""
Blend GCN scores with content scores:
  linear: s = alpha * s_gcn + (1 - alpha) * s_content
  rrf:    s = 1/(k + rank_gcn) + 1/(k + rank_cont)

Alpha:
  - constant list (e.g., 0.0,0.25,0.5,0.75,1.0)
  - dynamic per-item via degree: alpha = sigmoid((deg - t)/k)   (use --alphas dyn)

Score calibration per candidate set: --norm {none,z,minmax}

Examples:
  python -m scripts.eval_fusion --graph data/graph --ckpt runs/mmgcn/best.pt \
    --alphas 0.0,0.25,0.5,0.75,1.0 --norm z --cpu

  python -m scripts.eval_fusion --graph data/graph --ckpt runs/mmgcn/best.pt \
    --alphas dyn --alpha_t 3 --alpha_k 2 --extreme_cold_item --norm z --cpu

  # RRF（不依赖量纲）
  python -m scripts.eval_fusion --graph data/graph --ckpt runs/mmgcn/best.pt \
    --alphas 0.0 --fusion rrf --rrf_k 60 --cpu
"""
import argparse, os, json, time, csv, sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# 允许直接运行
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.mmgcn_light import MMGCN, SparseProp


def _calibrate(scores_np: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return scores_np
    if mode == "z":  # z-score per-candidate-set
        m = scores_np.mean()
        s = scores_np.std() + 1e-8
        return (scores_np - m) / s
    if mode == "minmax":
        lo = scores_np.min()
        hi = scores_np.max()
        return (scores_np - lo) / (hi - lo + 1e-8)
    raise ValueError(f"unknown norm mode: {mode}")


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


def build_user_profile(n_users, train_df, item_txt):
    prof = np.zeros((n_users, item_txt.shape[1]), dtype=np.float32)
    cnt  = np.zeros(n_users, dtype=np.int32)
    for u, i in zip(train_df["u"].to_numpy(), train_df["i"].to_numpy()):
        prof[u] += item_txt[i]
        cnt[u]  += 1
    cnt = np.maximum(cnt, 1)
    prof = prof / cnt[:, None]
    # cosine 归一化
    prof = prof / (np.linalg.norm(prof, axis=1, keepdims=True) + 1e-12)
    items = item_txt / (np.linalg.norm(item_txt, axis=1, keepdims=True) + 1e-12)
    return prof.astype(np.float32), items.astype(np.float32)


@torch.no_grad()
def evaluate_fusion(model, U, I, user_prof, item_txt, pairs_df, user_pos, n_items,
                    i_deg_arr, alpha_mode, alpha_const, t, k,
                    fusion_mode, rrf_k, norm_mode,
                    K=10, n_neg=99, device="cpu"):
    model.eval()
    HR, NDCG = [], []

    for u, i_pos in tqdm(pairs_df[["u", "i"]].to_numpy(), desc=f"Eval({alpha_mode}/{fusion_mode})", leave=False):
        # 负采样
        negs = set()
        while len(negs) < n_neg:
            j = np.random.randint(0, n_items)
            if (j not in user_pos[u]) and (j != i_pos):
                negs.add(j)
        cand = np.array([i_pos] + list(negs), dtype=np.int64)

        # GCN scores
        u_idx = torch.tensor([u] * len(cand), dtype=torch.long, device=device)
        i_idx = torch.tensor(cand, dtype=torch.long, device=device)
        s_gcn_np = model.score(U, I, u_idx, i_idx).detach().cpu().numpy()  # [C]

        # Content scores (cosine)
        s_cont_np = item_txt[cand] @ user_prof[u]  # [C]

        # 校准（逐候选集合）
        s_gcn_np  = _calibrate(s_gcn_np,  norm_mode)
        s_cont_np = _calibrate(s_cont_np, norm_mode)

        # alpha
        if alpha_mode == "const":
            a = float(alpha_const)
        else:  # dyn
            deg = i_deg_arr[cand]  # numpy
            a = 1.0 / (1.0 + np.exp(- (deg - t) / max(k, 1e-6)))

        # 融合
        if fusion_mode == "linear":
            s_np = a * s_gcn_np + (1.0 - a) * s_cont_np
        else:
            # Reciprocal Rank Fusion: 1/(k + rank)
            r_g = np.argsort(-s_gcn_np).argsort() + 1  # 1=best
            r_c = np.argsort(-s_cont_np).argsort() + 1
            s_np = 1.0 / (rrf_k + r_g) + 1.0 / (rrf_k + r_c)

        order = np.argsort(-s_np)
        rank  = int(np.where(order == 0)[0][0])  # 正样本在降序中的位置
        hit = 1 if rank < K else 0
        HR.append(hit)
        NDCG.append(1/np.log2(rank+2) if hit else 0.0)

    return float(np.mean(HR)), float(np.mean(NDCG))


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    meta, train_df, valid_df, test_df, feats, ui_edges = load_graph(args.graph)
    n_users, n_items = meta["n_users"], meta["n_items"]

    # 度数（训练集中 item 出现次数）
    i_cnt = train_df["i"].value_counts().reindex(range(n_items), fill_value=0).to_numpy()

    # 用户正集合
    user_pos = build_user_pos_set(n_users, train_df)

    # 内容向量
    user_prof, item_txt = build_user_profile(n_users, train_df, feats["txt"])

    # 模型与 U/I
    prop = SparseProp(ui_edges, n_users + n_items)
    x_txt = torch.tensor(feats["txt"],   dtype=torch.float32, device=device)
    x_img = torch.tensor(feats["img"],   dtype=torch.float32, device=device)
    x_st  = torch.tensor(feats["price"], dtype=torch.float32, device=device)
    ckpt = torch.load(args.ckpt, map_location=device)  # <-- 修正 map_location
    model = MMGCN(n_users, n_items, d=args.d, L=args.layers,
                  txt_dim=x_txt.shape[1], img_dim=x_img.shape[1], struct_dim=x_st.shape[1]).to(device)
    model.set_propagator(prop)
    model.load_state_dict(ckpt["state"], strict=False)
    with torch.no_grad():
        U, I = model.encode(x_txt, x_img, x_st, device)

    # 冷集定义
    u_cnt = train_df["u"].value_counts().reindex(range(n_users), fill_value=0)
    if args.extreme_cold_user:
        cold_users = set(u_cnt[u_cnt == 0].index.astype(int).tolist())
    else:
        cold_users = set(u_cnt[u_cnt <= args.cold_user_th].index.astype(int).tolist())
    if args.extreme_cold_item:
        cold_items = set(np.where(i_cnt == 0)[0].tolist())
    else:
        cold_items = set(np.where(i_cnt <= args.cold_item_th)[0].tolist())

    tasks = {
        "valid_all": valid_df,
        "test_all":  test_df,
        "valid_cold_user": valid_df[valid_df["u"].isin(cold_users)],
        "test_cold_user":  test_df[test_df["u"].isin(cold_users)],
        "valid_cold_item": valid_df[valid_df["i"].isin(cold_items)],
        "test_cold_item":  test_df[test_df["i"].isin(cold_items)],
    }

    # 解析 alphas
    alphas_raw = [s.strip() for s in args.alphas.split(",")]
    configs = []
    for a in alphas_raw:
        if a.lower() == "dyn":
            configs.append(("dyn", None))
        else:
            configs.append(("const", float(a)))

    # 评估 & 记录（统一 17 列表头，避免混档）
    out_csv = os.path.join(os.path.dirname(args.ckpt), "fusion_eval_log_v2.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    newfile = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if newfile:
            w.writerow(["time","alpha_mode","alpha_const","alpha_t","alpha_k",
                        "fusion","norm","split","n","HR","NDCG","K","n_neg",
                        "cold_user_th","cold_item_th","extreme_user","extreme_item"])
        for alpha_mode, aconst in configs:
            print(f"\n=== Alpha mode: {alpha_mode}  const={aconst}  fusion={args.fusion}  norm={args.norm}  (t={args.alpha_t}, k={args.alpha_k}) ===")
            for name, df in tasks.items():
                if len(df) == 0:
                    print(f"{name}: 0 samples, skip"); continue
                hr, nd = evaluate_fusion(
                    model, U, I, user_prof, item_txt, df, user_pos, n_items,
                    i_cnt, alpha_mode, (aconst if aconst is not None else 0.0),
                    args.alpha_t, args.alpha_k,
                    args.fusion, args.rrf_k, args.norm,
                    K=args.K, n_neg=args.n_neg, device=device
                )
                print(f"{name}: HR@{args.K}={hr:.4f}  NDCG@{args.K}={nd:.4f}  (n={len(df)})")
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                            alpha_mode, aconst, args.alpha_t, args.alpha_k,
                            args.fusion, args.norm,
                            name, len(df), hr, nd, args.K, args.n_neg,
                            args.cold_user_th, args.cold_item_th,
                            int(args.extreme_cold_user), int(args.extreme_cold_item)])
    print("Logged to:", out_csv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="data/graph")
    ap.add_argument("--ckpt",  default="runs/mmgcn/best.pt")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--n_neg", type=int, default=99)
    ap.add_argument("--cold_user_th", type=int, default=5)
    ap.add_argument("--cold_item_th", type=int, default=10)
    ap.add_argument("--extreme_cold_user", action="store_true")
    ap.add_argument("--extreme_cold_item", action="store_true")
    ap.add_argument("--alphas", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--alpha_t", type=float, default=3.0)   # degree threshold for dynamic alpha
    ap.add_argument("--alpha_k", type=float, default=2.0)   # softness for dynamic alpha
    ap.add_argument("--norm", default="z", choices=["none","z","minmax"])
    ap.add_argument("--fusion", default="linear", choices=["linear","rrf"])
    ap.add_argument("--rrf_k", type=float, default=60.0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    main(args)
