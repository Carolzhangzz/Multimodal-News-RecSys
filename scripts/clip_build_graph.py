# scripts/clip_build_graph.py
# -*- coding: utf-8 -*-
import argparse, os, json, re
import numpy as np
import pandas as pd
from tqdm import tqdm

def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"missing column '{c}' in df with columns: {list(df.columns)[:15]}...")

def build_index(series: pd.Series):
    if series.name is None:
        series = series.rename("id")
    uniq = series.drop_duplicates().reset_index(drop=True)
    mapping = {k: i for i, k in enumerate(uniq)}
    inv = pd.DataFrame({series.name: uniq, series.name + "_idx": np.arange(len(uniq))})
    return mapping, inv

def stack_vectors(series: pd.Series, dim: int, name: str):
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

def first_dim(series: pd.Series, default=0):
    for v in series:
        a = np.asarray(v, dtype=np.float32)
        if a.ndim == 1 and a.size > 0:
            return int(a.shape[0])
    return int(default)

def autodetect_cols(feats: pd.DataFrame):
    txt_col = next((c for c in feats.columns if re.match(r"^(clip_txt|txt_emb)_\d+$", c)), None)
    img_col = next((c for c in feats.columns if re.match(r"^(clip_img|img_emb)_\d+$", c)), None)
    if txt_col is None or img_col is None:
        raise KeyError(f"cannot find txt/img cols in features. got: {feats.columns[:15]}")
    if "price_log1p" not in feats.columns:
        feats["price_log1p"] = 0.0
    return txt_col, img_col, "price_log1p"

def make_img_mask(series: pd.Series):
    def nonzero(v):
        a = np.asarray(v, dtype=np.float32).ravel()
        return float(1.0 if (a.size > 0 and np.any(a != 0)) else 0.0)
    return series.map(nonzero).astype(np.float32).to_numpy().reshape(-1, 1)

def main(args):
    os.makedirs(args.out, exist_ok=True)

    inter = pd.read_parquet(args.interactions, engine="pyarrow")
    items = pd.read_parquet(args.items, engine="pyarrow")
    feats = pd.read_parquet(args.item_features, engine="pyarrow")

    ensure_cols(inter, ["user_id","item_id","ts"])
    ensure_cols(items, ["item_id"])
    ensure_cols(feats, ["item_id"])

    # 统一类型
    inter["user_id"] = inter["user_id"].astype(str)
    inter["item_id"] = inter["item_id"].astype(str)
    items["item_id"] = items["item_id"].astype(str)
    feats["item_id"] = feats["item_id"].astype(str)

    # 规范时间戳为秒
    if inter["ts"].dtype == "O":
        inter["ts"] = pd.to_datetime(inter["ts"], errors="coerce")
        inter["ts"] = inter["ts"].astype("int64") // 10**9
    inter["ts"] = pd.to_numeric(inter["ts"], errors="coerce").fillna(0).astype("int64")
    ms_mask = inter["ts"] > 10**12
    if ms_mask.any():
        inter.loc[ms_mask, "ts"] = inter.loc[ms_mask, "ts"] // 1000

    # 只保留有特征的 item
    items = items[items["item_id"].isin(set(feats["item_id"]))].copy()
    if len(items) == 0:
        raise ValueError("No items left after matching with features.item_id.")

    # 索引映射
    uid_map, uid_df = build_index(inter["user_id"])
    iid_map, iid_df = build_index(items["item_id"])

    # 过滤交互至有效 item
    inter = inter[inter["item_id"].isin(set(iid_map.keys()))].copy()
    inter["u"] = inter["user_id"].map(uid_map)
    inter["i"] = inter["item_id"].map(iid_map)
    inter = inter.dropna(subset=["u","i"])
    inter["u"] = inter["u"].astype(int)
    inter["i"] = inter["i"].astype(int)

    # 留一切分（不丢弃低历史用户，以便评测冷用户）
    train_u, train_i, val_u, val_i, test_u, test_i = [], [], [], [], [], []
    for u, dfu in tqdm(inter.sort_values("ts").groupby("user_id"), desc="split by user"):
        rows = dfu[["u","i","ts"]].sort_values("ts").to_numpy()
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

    # 保存 split 与索引
    uid_df.to_parquet(f"{args.out}/users_idx.parquet", index=False)
    iid_df.to_parquet(f"{args.out}/items_idx.parquet", index=False)
    train.to_parquet(f"{args.out}/train_edges.parquet", index=False)
    valid.to_parquet(f"{args.out}/valid_pairs.parquet", index=False)
    test .to_parquet(f"{args.out}/test_pairs.parquet",  index=False)

    # 训练度数（可用于分析）
    train_user_deg = train.groupby("u").size().reset_index(name="deg")
    train_item_deg = train.groupby("i").size().reset_index(name="deg")
    train_user_deg.to_parquet(f"{args.out}/train_user_deg.parquet", index=False)
    train_item_deg.to_parquet(f"{args.out}/train_item_deg.parquet", index=False)

    # 对齐 item 特征 → npz（含 img_mask）
    feats = feats.merge(iid_df, on="item_id", how="inner").sort_values("item_id_idx").reset_index(drop=True)
    txt_col, img_col, price_col = autodetect_cols(feats)
    txt_dim   = first_dim(feats[txt_col], 384)
    img_dim   = first_dim(feats[img_col], 512)

    txt   = stack_vectors(feats[txt_col],   txt_dim,   txt_col)
    img   = stack_vectors(feats[img_col],   img_dim,   img_col)
    price = feats[price_col].astype(np.float32).to_numpy().reshape(-1, 1)

    if "img_mask" in feats.columns:
        mask = feats["img_mask"].astype(np.float32).to_numpy().reshape(-1,1)
    else:
        mask = make_img_mask(feats[img_col])

    np.savez_compressed(f"{args.out}/item_features_aligned.npz",
                        txt=txt, img=img, price=price, img_mask=mask)

    meta = {
        "n_users": int(len(uid_df)),
        "n_items": int(len(iid_df)),
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "n_test":  int(len(test)),
        "txt_dim": int(txt_dim),
        "img_dim": int(img_dim),
        "struct_dim": 1
    }
    with open(f"{args.out}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved to", args.out, meta)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions", default="data/processed/interactions.parquet")
    ap.add_argument("--items", default="data/processed/items.parquet")
    ap.add_argument("--item_features", default="data/features/item_features_clip.parquet")
    ap.add_argument("--out", default="data/graph_full")
    args = ap.parse_args()
    main(args)
