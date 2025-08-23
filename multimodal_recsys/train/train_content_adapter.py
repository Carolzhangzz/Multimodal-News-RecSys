# train/train_content_adapter.py
# -*- coding: utf-8 -*-

# 输入：物品文本向量 txt(384) + 价格(1) → 385 维

# 思路：学两层线性投影（用户/物品各一层），用 BPR 训练

# 特点：不学 ID embedding，可直接泛化到训练没出现过的物品
import argparse, os, json, numpy as np, pandas as pd
import torch, torch.nn as nn
from tqdm import tqdm

SEED=2025
torch.manual_seed(SEED); np.random.seed(SEED)

def load_data(graph_dir):
    with open(os.path.join(graph_dir, "meta.json")) as f:
        meta = json.load(f)
    train = pd.read_parquet(os.path.join(graph_dir, "train_edges.parquet"))
    valid = pd.read_parquet(os.path.join(graph_dir, "valid_pairs.parquet"))
    test  = pd.read_parquet(os.path.join(graph_dir, "test_pairs.parquet"))
    feats = np.load(os.path.join(graph_dir, "item_features_aligned.npz"))
    return meta, train, valid, test, feats

def build_user_profiles(n_users, train_df, item_txt, item_price):
    X = np.concatenate([item_txt, item_price], axis=1)  # [n_items, 385]
    prof = np.zeros((n_users, X.shape[1]), dtype=np.float32)
    cnt  = np.zeros(n_users, dtype=np.int32)
    for u,i in zip(train_df["u"].to_numpy(), train_df["i"].to_numpy()):
        prof[u] += X[i]; cnt[u]+=1
    cnt = np.maximum(cnt, 1)
    prof = prof / cnt[:,None]
    return prof, X

class ContentAdapter(nn.Module):
    def __init__(self, in_dim=385, d=128):
        super().__init__()
        self.user_lin = nn.Linear(in_dim, d, bias=True)
        self.item_lin = nn.Linear(in_dim, d, bias=True)
    def forward(self, U_raw, I_raw, u_idx, i_idx):
        U = self.user_lin(U_raw)   # [n_users, d]
        I = self.item_lin(I_raw)   # [n_items, d]
        return (U[u_idx] * I[i_idx]).sum(-1), U, I

class BPRLoss(nn.Module):
    def forward(self, pos, neg):
        return -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()

def sample_neg(n_items, user_pos, u_batch):
    neg=[]
    for u in u_batch:
        s=user_pos[u]; j=np.random.randint(0,n_items)
        while j in s: j=np.random.randint(0,n_items)
        neg.append(j)
    return np.array(neg, dtype=np.int64)

@torch.no_grad()
def evaluate(U, I, pairs_df, user_pos, n_items, K=10, n_neg=99, device="cpu"):
    HR,NDCG=[],[]
    for u,i_pos in tqdm(pairs_df[["u","i"]].to_numpy(), desc="Eval", leave=False):
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
    os.makedirs(args.out, exist_ok=True)
    meta, train_df, valid_df, test_df, feats = load_data(args.graph)
    n_users, n_items = meta["n_users"], meta["n_items"]

    U_raw_np, I_raw_np = build_user_profiles(n_users, train_df, feats["txt"], feats["price"])
    U_raw = torch.tensor(U_raw_np, dtype=torch.float32, device=device)
    I_raw = torch.tensor(I_raw_np, dtype=torch.float32, device=device)

    user_pos=[set() for _ in range(n_users)]
    for u,i in zip(train_df["u"].to_numpy(), train_df["i"].to_numpy()):
        user_pos[int(u)].add(int(i))

    model=ContentAdapter(in_dim=U_raw.shape[1], d=args.d).to(device)
    opt=torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit=BPRLoss()

    u_arr=train_df["u"].to_numpy(np.int64); i_arr=train_df["i"].to_numpy(np.int64)
    best=-1.0
    for ep in range(1, args.epochs+1):
        idx=np.random.permutation(len(u_arr))
        u_s, i_s = u_arr[idx], i_arr[idx]
        losses=[]
        for s in range(0, len(u_s), args.batch):
            ub=u_s[s:s+args.batch]; ib=i_s[s:s+args.batch]; jb=sample_neg(n_items, user_pos, ub)
            u_t=torch.tensor(ub, dtype=torch.long, device=device)
            i_t=torch.tensor(ib, dtype=torch.long, device=device)
            j_t=torch.tensor(jb, dtype=torch.long, device=device)
            pos,_,_ = model(U_raw, I_raw, u_t, i_t)
            neg,_,_ = model(U_raw, I_raw, u_t, j_t)
            loss=crit(pos, neg)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        with torch.no_grad():
            _, U, I = model(U_raw, I_raw,
                            torch.tensor([0],dtype=torch.long,device=device),
                            torch.tensor([0],dtype=torch.long,device=device))
            hr, nd = evaluate(U, I, valid_df, user_pos, n_items, K=10, n_neg=99, device=device)
        print(f"Epoch {ep:02d} | loss {np.mean(losses):.4f} | valid HR@10 {hr:.4f} NDCG@10 {nd:.4f}")
        if hr+nd>best:
            best=hr+nd
            torch.save({"state":model.state_dict(),
                        "meta":meta, "d":args.d, "in_dim":int(U_raw.shape[1])},
                       os.path.join(args.out,"best.pt"))

    ckpt=torch.load(os.path.join(args.out,"best.pt"), map_location=device)
    model.load_state_dict(ckpt["state"])
    with torch.no_grad():
        _, U, I = model(U_raw, I_raw,
                        torch.tensor([0],dtype=torch.long,device=device),
                        torch.tensor([0],dtype=torch.long,device=device))
        hr, nd = evaluate(U, I, test_df, user_pos, n_items, K=10, n_neg=99, device=device)
    print(f"TEST HR@10 {hr:.4f} NDCG@10 {nd:.4f}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--graph", default="data/graph")
    ap.add_argument("--out",   default="runs/content_adapter")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch",  type=int, default=4096)
    ap.add_argument("--d",      type=int, default=128)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--wd",     type=float, default=1e-4)
    ap.add_argument("--cpu", action="store_true")
    args=ap.parse_args()
    main(args)
