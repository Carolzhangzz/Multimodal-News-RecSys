# mmgcn_train.py
# -*- coding: utf-8 -*-
import argparse, os, json, math, random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------- Model -------
class SparseProp(nn.Module):
    def __init__(self, edge_index: torch.Tensor, num_nodes: int, device=None):
        super().__init__()
        if device is None: device = edge_index.device
        idx = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # symmetric
        deg = torch.bincount(idx[0], minlength=num_nodes).float().clamp(min=1)
        di = deg[idx[0]].rsqrt()
        dj = deg[idx[1]].rsqrt()
        val = di * dj
        self.register_buffer("A_idx", idx.to(device))
        self.register_buffer("A_val", val.to(device))
        self.num_nodes = num_nodes
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = torch.sparse_coo_tensor(self.A_idx, self.A_val,
                                    (self.num_nodes, self.num_nodes),
                                    device=x.device)
        return torch.sparse.mm(A, x)

class MMGCN(nn.Module):
    def __init__(self, n_users, n_items, d, L, txt_dim, img_dim, struct_dim):
        super().__init__()
        self.n_users, self.n_items, self.d, self.L = n_users, n_items, d, L

        def branch(_txt, _img, _st):
            return nn.ModuleDict({
                "user": nn.Embedding(n_users, d),
                "proj_txt": nn.Linear(_txt, d),
                "proj_img": nn.Linear(_img, d),
                "proj_struct": nn.Linear(_st, d),
            })
        self.br_txt = branch(txt_dim, img_dim, struct_dim)
        self.br_img = branch(txt_dim, img_dim, struct_dim)
        self.br_struct = branch(txt_dim, img_dim, struct_dim)

        self.attn_u = nn.Sequential(nn.Linear(3*d, 128), nn.ReLU(), nn.Linear(128, 3))
        self.attn_i = nn.Sequential(nn.Linear(3*d, 128), nn.ReLU(), nn.Linear(128, 3))

        self.prop = None

    def set_propagator(self, prop: SparseProp):
        self.prop = prop

    def _run_branch(self, br, x_txt, x_img, x_struct, device, item_img_mask=None):
        # 初始节点
        u0 = br["user"].weight                           # [n_users, d]
        i_txt = br["proj_txt"](x_txt)                    # [n_items, d]
        i_img = br["proj_img"](x_img)                    # [n_items, d]
        i_st  = br["proj_struct"](x_struct)              # [n_items, d]

        if item_img_mask is not None:
            i_img = i_img * item_img_mask               # 幅度门控

        x0 = torch.cat([u0, i_txt + i_img + i_st], dim=0).to(device)

        out = x0
        for _ in range(self.L):
            x0 = self.prop(x0)
            out = out + x0
        out = out / (self.L + 1)

        U, I = out[:self.n_users], out[self.n_users:]
        return U, I, i_img  # 返回 i_img 供 item 侧注意力屏蔽（logit级）

    def encode(self, x_txt, x_img, x_struct, device, item_img_mask=None):
        U_txt, I_txt, _      = self._run_branch(self.br_txt,    x_txt, x_img, x_struct, device, item_img_mask)
        U_img, I_img, I_imgp = self._run_branch(self.br_img,    x_txt, x_img, x_struct, device, item_img_mask)
        U_st , I_st , _      = self._run_branch(self.br_struct, x_txt, x_img, x_struct, device, item_img_mask)

        U_cat = torch.cat([U_txt, U_img, U_st], dim=1)
        I_cat = torch.cat([I_txt, I_img, I_st], dim=1)

        a_u = torch.softmax(self.attn_u(U_cat), dim=-1)  # [n_users,3]
        logits_i = self.attn_i(I_cat)                    # [n_items,3]

        if item_img_mask is not None:
            # mask=0 → 压低图像通道 logit
            neg_inf = (1.0 - item_img_mask) * -1e4
            # 通道顺序: [txt, img, struct] —— 仅对 img 通道加偏置
            bias = torch.cat([torch.zeros_like(neg_inf), neg_inf, torch.zeros_like(neg_inf)], dim=1)
            logits_i = logits_i + bias

        a_i = torch.softmax(logits_i, dim=-1)

        U = a_u[:,0:1]*U_txt + a_u[:,1:2]*U_img + a_u[:,2:3]*U_st
        I = a_i[:,0:1]*I_txt + a_i[:,1:2]*I_img + a_i[:,2:3]*I_st
        return U, I

    def score(self, U, I, u_idx, i_idx):
        return (U[u_idx] * I[i_idx]).sum(-1)

# ------- utils -------
def bpr_loss(pos, neg):
    return -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()

@torch.no_grad()
def recall_ndcg_at_k(U, I, pairs, user_pos, K=20, n_neg=1000, device="cpu"):
    HR, NDCG = [], []
    rng = np.random.default_rng(2025)
    n_items = I.shape[0]
    for u, i_pos in pairs:
        negs = set()
        while len(negs) < n_neg:
            j = int(rng.integers(0, n_items))
            if (j not in user_pos[u]) and (j != i_pos):
                negs.add(j)
        cand = torch.tensor([i_pos] + list(negs), dtype=torch.long, device=device)
        u_idx = torch.tensor([u]*cand.numel(), dtype=torch.long, device=device)
        scores = (U[u_idx] * I[cand]).sum(-1)
        order = torch.argsort(scores, descending=True)
        rank = int((order == 0).nonzero(as_tuple=True)[0].item())
        hit = 1 if rank < K else 0
        HR.append(hit)
        NDCG.append(1/np.log2(rank+2) if hit else 0.0)
    return float(np.mean(HR)), float(np.mean(NDCG))

def load_graph(graph_dir, device):
    with open(os.path.join(graph_dir, "meta.json")) as f:
        meta = json.load(f)
    train = pd.read_parquet(os.path.join(graph_dir, "train_edges.parquet"))
    valid = pd.read_parquet(os.path.join(graph_dir, "valid_pairs.parquet"))
    test  = pd.read_parquet(os.path.join(graph_dir, "test_pairs.parquet"))
    feats = np.load(os.path.join(graph_dir, "item_features_aligned.npz"))
    n_users, n_items = meta["n_users"], meta["n_items"]

    ui = torch.tensor(
        np.stack([train["u"].to_numpy(), train["i"].to_numpy() + n_users], axis=0),
        dtype=torch.long, device=device
    )
    prop = SparseProp(ui, n_users + n_items, device=device)

    x_txt = torch.tensor(feats["txt"],      dtype=torch.float32, device=device)
    x_img = torch.tensor(feats["img"],      dtype=torch.float32, device=device)
    x_st  = torch.tensor(feats["price"],    dtype=torch.float32, device=device)
    img_mask = torch.tensor(feats["img_mask"], dtype=torch.float32, device=device) \
               if "img_mask" in feats.files else None

    return meta, train, valid, test, prop, x_txt, x_img, x_st, img_mask

def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    meta, train_df, valid_df, test_df, prop, x_txt, x_img, x_st, img_mask = load_graph(args.graph_dir, device)
    n_users, n_items = meta["n_users"], meta["n_items"]

    model = MMGCN(n_users, n_items, d=args.dim, L=args.layers,
                  txt_dim=x_txt.shape[1], img_dim=x_img.shape[1], struct_dim=x_st.shape[1]).to(device)
    model.set_propagator(prop)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # build pos set
    user_pos = [set() for _ in range(n_users)]
    for u,i in zip(train_df["u"].to_numpy(), train_df["i"].to_numpy()):
        user_pos[int(u)].add(int(i))

    # training pairs
    train_pairs = list(zip(train_df["u"].tolist(), train_df["i"].tolist()))
    steps_per_epoch = math.ceil(len(train_pairs) / args.batch_size)

    for epoch in range(1, args.epochs+1):
        model.train()
        random.shuffle(train_pairs)
        losses = []
        for bi in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}"):
            batch = train_pairs[bi*args.batch_size : (bi+1)*args.batch_size]
            if not batch: break
            u = torch.tensor([x[0] for x in batch], dtype=torch.long, device=device)
            i = torch.tensor([x[1] for x in batch], dtype=torch.long, device=device)
            # negative sampling
            j = torch.randint(0, n_items, (len(batch),), device=device)
            # avoid sampling positives (simple but effective)
            for t in range(len(batch)):
                while int(j[t].item()) in user_pos[int(u[t].item())]:
                    j[t] = torch.randint(0, n_items, (1,), device=device)

            U, I = model.encode(x_txt, x_img, x_st, device=device, item_img_mask=img_mask)
            pos = model.score(U, I, u, i)
            neg = model.score(U, I, u, j)
            loss = bpr_loss(pos, neg)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # ---- validation ----
        model.eval()
        with torch.no_grad():
            U_eval, I_eval = model.encode(x_txt, x_img, x_st, device=device, item_img_mask=img_mask)
        val_pairs = list(zip(valid_df["u"].tolist(), valid_df["i"].tolist()))
        r, nd = recall_ndcg_at_k(U_eval, I_eval, val_pairs, user_pos,
                                 K=20, n_neg=args.eval_neg, device=device)
        print(f"[Epoch {epoch}] train_loss={np.mean(losses):.4f} | val@20: recall={r:.4f}, ndcg={nd:.4f}")

        # early stop
        if epoch == 1:
            best, best_epoch = nd, epoch
            torch.save({"model": model.state_dict(),
                        "meta": {"dim": args.dim, "layers": args.layers}}, os.path.join(args.out, "mmgcn_light.pt"))
        else:
            if nd > best + 1e-4:
                best, best_epoch = nd, epoch
                torch.save({"model": model.state_dict(),
                            "meta": {"dim": args.dim, "layers": args.layers}}, os.path.join(args.out, "mmgcn_light.pt"))
            elif epoch - best_epoch >= args.patience:
                print("Early stop.")
                break

    # final test (整体)
    ckpt = torch.load(os.path.join(args.out, "mmgcn_light.pt"), map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    with torch.no_grad():
        U_eval, I_eval = model.encode(x_txt, x_img, x_st, device=device, item_img_mask=img_mask)
    test_pairs = list(zip(test_df["u"].tolist(), test_df["i"].tolist()))
    r, nd = recall_ndcg_at_k(U_eval, I_eval, test_pairs, user_pos, K=20, n_neg=args.eval_neg, device=device)
    print(f"[Test] recall@20={r:.4f}, ndcg@20={nd:.4f}")
    print("Saved checkpoint to:", os.path.join(args.out, "mmgcn_light.pt"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--eval-neg", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    main(args)
