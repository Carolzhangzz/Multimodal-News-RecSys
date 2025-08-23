import argparse, os, io, requests, math
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from sentence_transformers import SentenceTransformer

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def to_plain_text(x):
    """把 list/dict/NaN 统一转成可读字符串，避免 SentenceTransformer 报错。"""
    if isinstance(x, list):
        return " ".join(str(t) for t in x if t)
    if isinstance(x, dict):
        return " ".join(str(v) for v in x.values() if v)
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x)

def download_first_image(url_list, save_dir, item_id, timeout=8):
    if not isinstance(url_list, list) or len(url_list) == 0:
        return None
    url = url_list[0]
    fn = os.path.join(save_dir, f"{item_id}.jpg")
    if os.path.exists(fn):
        return fn
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        Image.open(io.BytesIO(r.content)).convert("RGB").save(fn, format="JPEG")
        return fn
    except Exception:
        return None

def encode_text(texts, batch_size=256, device="cpu"):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = []
    # sbert 内部也有 batch；外层 batch 主要控内存
    for i in tqdm(range(0, len(texts), batch_size), desc="Text emb"):
        batch = texts[i:i+batch_size]
        e = model.encode(
            batch,
            device=device,
            batch_size=min(batch_size, 64),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        embs.append(e)
    return np.vstack(embs) if embs else np.zeros((0,384), dtype=np.float32)

def encode_images(paths, batch_size=64, device="cpu"):
    # 只有需要图像时才导入 open_clip，避免无谓依赖开销
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    model.eval()
    out_dim = model.visual.output_dim
    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Image emb"):
        imgs = []
        for p in paths[i:i+batch_size]:
            if p is None:
                imgs.append(torch.zeros(3,224,224))
            else:
                try:
                    img = Image.open(p).convert("RGB")
                    imgs.append(preprocess(img))
                except Exception:
                    imgs.append(torch.zeros(3,224,224))
        tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            f = model.encode_image(tensor)
            f = f / f.norm(dim=1, keepdim=True).clamp(min=1e-8)
        feats.append(f.cpu().numpy().astype(np.float32))
    return np.vstack(feats) if feats else np.zeros((len(paths), out_dim), dtype=np.float32)

def main(args):
    ensure_dir(args.out); ensure_dir(args.img_dir)

    items = pd.read_parquet(args.items, engine="pyarrow")
    inter = pd.read_parquet(args.interactions, engine="pyarrow")

    # 仅保留在 interactions 出现过的 item，可大幅减少计算量
    if args.only_interacted:
        keep = set(inter["item_id"].unique())
        items = items[items["item_id"].isin(keep)].copy()

    # 进一步限量（快速验流程）
    if args.limit_items > 0:
        items = items.head(args.limit_items).copy()

    # 文本拼接（把 list/dict 统一转纯文本）
    title = items["title"].apply(to_plain_text)
    desc  = items["description"].apply(to_plain_text)
    cat   = items.get("category", pd.Series([""]*len(items))).apply(to_plain_text)
    text  = (title + " [SEP] " + desc + " [CAT] " + cat).str.slice(0, 800).tolist()

    # 1) 文本特征
    txt_emb = encode_text(text, batch_size=args.txt_bs, device="cpu")  # [N, 384]

    # 2) 图像特征（可跳过）
    if args.no_image:
        img_emb = np.zeros((len(items), 512), dtype=np.float32)
        has_image_flags = [False]*len(items)
    else:
        # 如无 image_urls 列，补空
        if "image_urls" not in items.columns:
            items["image_urls"] = [[] for _ in range(len(items))]
        img_paths = []
        for iid, urls in tqdm(zip(items["item_id"], items["image_urls"]), total=len(items), desc="Download imgs"):
            img_paths.append(download_first_image(urls, args.img_dir, iid))
        has_image_flags = [p is not None for p in img_paths]
        img_emb = encode_images(img_paths, batch_size=args.img_bs, device="cpu")  # [N, 512]

    # 3) 结构化：price
    price_series = items.get("price", pd.Series([0.0]*len(items))).astype(float).fillna(0.0).clip(lower=0.0)
    price = np.log1p(price_series.to_numpy(dtype=np.float32))  # shape: [N]

    # 保存
    out_df = pd.DataFrame({
        "item_id": items["item_id"].values,
        "txt_emb_384": [v.tolist() for v in txt_emb],
        "img_emb_512": [v.tolist() for v in img_emb],
        "price_log1p": price.tolist(),
        "has_image": has_image_flags,
    })
    out_df.to_parquet(f"{args.out}/item_features.parquet", engine="pyarrow", index=False)

    # 用户统计特征
    g = inter.groupby("user_id")
    user_stats = pd.DataFrame({
        "user_id": g.size().index,
        "hist_len": g.size().values,
        "mean_rating": g["rating"].mean().fillna(0).values,
        "last_ts": g["ts"].max().values,
    })
    user_stats.to_parquet(f"{args.out}/user_stats.parquet", engine="pyarrow", index=False)

    print("Saved:", f"{args.out}/item_features.parquet", "and", f"{args.out}/user_stats.parquet")
    print("items encoded:", len(items))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/processed/items.parquet")
    ap.add_argument("--interactions", default="data/processed/interactions.parquet")
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--img_dir", default="data/images")
    ap.add_argument("--no-image", action="store_true")
    ap.add_argument("--only-interacted", action="store_true", help="仅对出现在 interactions 中的 item 抽特征")
    ap.add_argument("--limit-items", type=int, default=0, help="最多处理多少个 item（0 表示不限制）")
    ap.add_argument("--txt-bs", type=int, default=256)
    ap.add_argument("--img-bs", type=int, default=64)
    args = ap.parse_args()
    main(args)
