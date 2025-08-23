# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class SparseProp(nn.Module):
    """
    LightGCN 风格：x^{l+1} = \\hat{A} x^l，\\hat{A} 为对称归一化邻接。
    用 torch.sparse_coo_tensor 做稀疏乘法，CPU/GPU 通用。
    """
    def __init__(self, edge_index: torch.Tensor, num_nodes: int):
        super().__init__()
        # edge_index: [2, E] (users+items 的同构图坐标；物品索引需提前 +n_users)
        idx = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # 对称
        deg = torch.bincount(idx[0], minlength=num_nodes).float().clamp(min=1)
        di = deg[idx[0]].rsqrt()
        dj = deg[idx[1]].rsqrt()
        val = di * dj
        self.register_buffer("A_idx", idx)
        self.register_buffer("A_val", val)
        self.num_nodes = num_nodes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 确保索引和值在同一 device
        idx = self.A_idx.to(x.device)
        val = self.A_val.to(x.device)
        A = torch.sparse_coo_tensor(idx, val, (self.num_nodes, self.num_nodes), device=x.device)
        return torch.sparse.mm(A, x)

class MMGCN(nn.Module):
    def __init__(self, n_users, n_items, d, L, txt_dim, img_dim, struct_dim):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.d = d
        self.L = L

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

        self.prop = None  # 训练脚本里 set_propagator

    def set_propagator(self, prop: SparseProp):
        self.prop = prop

    def _run_branch(self, br, x_txt, x_img, x_struct, device):
        # 初始节点 (n_users+n_items, d)
        u0 = br["user"].weight                      # [n_users, d]
        i0 = br["proj_txt"](x_txt) + br["proj_img"](x_img) + br["proj_struct"](x_struct)  # [n_items, d]
        x = torch.cat([u0, i0], dim=0).to(device)

        out = x  # 包含第0层
        for _ in range(self.L):
            x = self.prop(x)
            out = out + x
        out = out / (self.L + 1)

        U, I = out[:self.n_users], out[self.n_users:]
        return U, I

    def encode(self, x_txt, x_img, x_struct, device):
        U_txt, I_txt = self._run_branch(self.br_txt, x_txt, x_img, x_struct, device)
        U_img, I_img = self._run_branch(self.br_img, x_txt, x_img, x_struct, device)
        U_st , I_st  = self._run_branch(self.br_struct, x_txt, x_img, x_struct, device)

        U_cat = torch.cat([U_txt, U_img, U_st], dim=1)
        I_cat = torch.cat([I_txt, I_img, I_st], dim=1)
        a_u = torch.softmax(self.attn_u(U_cat), dim=-1)  # [n_users,3]
        a_i = torch.softmax(self.attn_i(I_cat), dim=-1)  # [n_items,3]

        U = a_u[:, 0:1]*U_txt + a_u[:, 1:2]*U_img + a_u[:, 2:3]*U_st
        I = a_i[:, 0:1]*I_txt + a_i[:, 1:2]*I_img + a_i[:, 2:3]*I_st
        return U, I

    def score(self, U, I, u_idx, i_idx):
        return (U[u_idx] * I[i_idx]).sum(-1)
