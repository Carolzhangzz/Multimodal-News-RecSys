# scripts/03_build_graph.py
# -*- coding: utf-8 -*-
import argparse, os, json, sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---- utils ----
def build_index(series: pd.Series):
    if series.name is None:
        series = series.rename("id")
    uniq = series.drop_duplicates().reset_index(drop=True)
    mapping = {k: i for i, k in enumerate(uniq)}
    inv = pd.DataFrame({series.name: uniq, series.name + "_idx": np.arange(len(uniq))})
    return mapping, inv

def stack_vectors(series: pd.Series, dim: int, name: str):
    """把 list/array 组成的列安全堆叠成 [N, dim]；维度不符/缺失补 0，并统计个数。"""
    arrs, bad = [], 0
    for v in series:
        a = np.asarray(v, dtype=np.float32)
        if a.ndim != 1 or a.shape[0] != dim:
            bad += 1
            a = np.zeros((dim,), dtype=np.float32)
        arrs.append(a)
    if bad > 0:
        print(f"[warn] {name}: {bad} vectors had wrong shape; filled with zeros.")
    return np.stack(arrs, axis=0)

def ensure_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"missing column '{c}' in dataframe with columns: {list(df.columns)[:10]}...")

# ---- main ----
def main(args):
    os.makedirs(args.out, exist_ok=True)

    # 读取
    inter = pd.read_parquet(args.interactions, engine="pyarrow")
    items = pd.read_parquet(args.items, engine="pyarrow")
    feats = pd.read_parquet(args.item_features, engine="pyarrow")

    # 列存在性检查
    ensure_cols(inter, ["user_id", "item_id", "ts"])
    ensure_cols(items, ["item_id"])
    ensure_cols(feats, ["item_id", "txt_emb_384", "img_emb_512", "price_log1p"])

    # 只保留有特征的 item
    items = items[items["item_id"].isin(set(feats["item_id"]))].copy()
    if len(items) == 0:
        raise ValueError("No items left after filtering by item_features.item_id. "
                         "Check that item_features.parquet 与 items.parquet 的 item_id 一致。")

    # 索引映射
    uid_map, uid_df = build_index(inter["user_id"])
    iid_map, iid_df = build_index(items["item_id"])

    # 过滤交互：只保留存在于 iid_map 的 item
    inter = inter[inter["item_id"].isin(set(iid_map.keys()))].copy()

    # 添加索引列
    inter["u"] = inter["user_id"].map(uid_map)
    inter["i"] = inter["item_id"].map(iid_map)
    inter = inter.dropna(subset=["u", "i"])

    # ts 可能是毫秒/字符串，统一为 int 秒（仅用于排序）
    if inter["ts"].dtype == "O":
        inter["ts"] = pd.to_datetime(inter["ts"], errors="coerce")
        inter["ts"] = inter["ts"].astype("int64") // 10**9
    inter["ts"] = pd.to_numeric(inter["ts"], errors="coerce").fillna(0).astype("int64")
    # 毫秒转秒
    mask_ms = inter["ts"] > 10**12
    if mask_ms.any():
        inter.loc[mask_ms, "ts"] = inter.loc[mask_ms, "ts"] // 1000

    # 按用户分组切分（留一法）
    train_u, train_i, val_u, val_i, test_u, test_i = [], [], [], [], [], []
    for u, dfu in tqdm(inter.sort_values("ts").groupby("user_id"), desc="split by user"):
        rows = dfu[["u", "i", "ts"]].sort_values("ts").to_numpy()
        if len(rows) == 1:
            test_u.append(int(rows[-1][0])); test_i.append(int(rows[-1][1]))
        elif len(rows) == 2:
            train_u.append(int(rows[0][0])); train_i.append(int(rows[0][1]))
            test_u.append(int(rows[1][0]));  test_i.append(int(rows[1][1]))
        else:
            val_u.append(int(rows[-2][0]));  val_i.append(int(rows[-2][1]))
            test_u.append(int(rows[-1][0])); test_i.append(int(rows[-1][1]))
            for r in rows[:-2]:
                train_u.append(int(r[0])); train_i.append(int(r[1]))

    train = pd.DataFrame({"u": train_u, "i": train_i})
    valid = pd.DataFrame({"u": val_u, "i": val_i})
    test  = pd.DataFrame({"u": test_u, "i": test_i})

    # 保存索引映射与切分
    uid_df.to_parquet(f"{args.out}/users_idx.parquet", index=False, engine="pyarrow")
    iid_df.to_parquet(f"{args.out}/items_idx.parquet", index=False, engine="pyarrow")
    train.to_parquet(f"{args.out}/train_edges.parquet", index=False, engine="pyarrow")
    valid.to_parquet(f"{args.out}/valid_pairs.parquet", index=False, engine="pyarrow")
    test.to_parquet(f"{args.out}/test_pairs.parquet", index=False, engine="pyarrow")

    # 对齐并导出 item 特征为 npz（带兜底）
    feats = feats.merge(iid_df, on="item_id", how="inner").sort_values("item_id_idx").reset_index(drop=True)
    txt   = stack_vectors(feats["txt_emb_384"], 384, "txt_emb_384")
    img   = stack_vectors(feats["img_emb_512"], 512, "img_emb_512")
    price = feats["price_log1p"].astype(np.float32).to_numpy().reshape(-1, 1)

    np.savez_compressed(f"{args.out}/item_features_aligned.npz", txt=txt, img=img, price=price)

    meta = {
        "n_users": int(len(uid_df)),
        "n_items": int(len(iid_df)),
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "n_test":  int(len(test)),
    }
    with open(f"{args.out}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved to", args.out, meta)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions", default="data/processed/interactions.parquet")
    ap.add_argument("--items", default="data/processed/items.parquet")
    ap.add_argument("--item_features", default="data/features/item_features.parquet")
    ap.add_argument("--out", default="data/graph")
    args = ap.parse_args()
    main(args)
