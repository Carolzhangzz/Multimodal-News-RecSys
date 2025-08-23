import argparse, ast, json, os, gzip
import pandas as pd

def load_any(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path, engine="pyarrow")
    opener = gzip.open if path.endswith(".gz") else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)

def pick(df, names, default=None):
    for n in names:
        if n in df.columns:
            return df[n]
    return default

def to_list(x):
    if isinstance(x, list): return x
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            return v if isinstance(v, list) else [x]
        except Exception:
            return [x]
    return [] if pd.isna(x) else [x]

def extract_image_urls(img_field):
    urls = []
    for it in to_list(img_field):
        if isinstance(it, dict):
            for k in ("hi_res","large","thumb"):
                u = it.get(k)
                if isinstance(u, str): urls.append(u)
        elif isinstance(it, str):
            urls.append(it)
    # 去重保序
    seen=set(); out=[]
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out

def main(args):
    print("==> Loading")
    reviews = load_any(args.reviews)
    meta    = load_any(args.meta)

    # ---------- items ----------
    asin        = pick(meta, ["asin","product_id","item_id"])
    parent_asin = pick(meta, ["parent_asin","parentAsin"])
    title       = pick(meta, ["title","item_title","name"])
    desc        = pick(meta, ["description","item_description","desc"])
    price       = pick(meta, ["price"])
    category    = pick(meta, ["categories","category_path","category","main_category"])
    images      = pick(meta, ["images","image_urls","imageURL","imageURLHighRes"])

    items = pd.DataFrame({
        "asin": asin,
        "parent_asin": parent_asin if parent_asin is not None else asin,
        "title": title, "description": desc, "price": price,
        "category": category, "images_raw": images
    })

    # 类目转文本
    def cat_str(x):
        if isinstance(x, list):
            flat=[]
            for e in x:
                if isinstance(e, list): flat += e
                else: flat.append(e)
            flat = [str(t) for t in flat if t]
            return " > ".join(flat) if flat else None
        return str(x) if pd.notna(x) else None
    items["category"] = items["category"].apply(cat_str)

    # 图片 URL 列表
    items["image_urls"] = items["images_raw"].apply(extract_image_urls)

    # item_id = parent_asin（缺失则用 asin）
    items["item_id"] = items["parent_asin"].fillna(items["asin"])
    items = items.dropna(subset=["item_id"]).drop_duplicates(subset=["item_id"])

    # ---------- interactions ----------
    user_id   = pick(reviews, ["user_id","reviewerID","reviewer_id","customer_id"])
    asin_r    = pick(reviews, ["parent_asin","asin","product_id","item_id"])
    rating    = pick(reviews, ["rating","overall","star_rating"])
    ts        = pick(reviews, ["timestamp","sort_timestamp","unixReviewTime","reviewTime","time"])
    rev_title = pick(reviews, ["title","summary"])
    rev_text  = pick(reviews, ["text","reviewText","content"])

    inter = pd.DataFrame({
        "user_id": user_id, "asin_or_parent": asin_r, "rating": rating, "ts": ts,
        "rev_title": rev_title, "rev_text": rev_text
    }).dropna(subset=["user_id","asin_or_parent"])

    # map 到 item_id（优先 parent_asin）
    asin2item   = items[["asin","item_id"]].dropna().drop_duplicates().rename(columns={"asin":"asin_or_parent"})
    parent2item = items[["parent_asin","item_id"]].dropna().drop_duplicates().rename(columns={"parent_asin":"asin_or_parent"})
    map_df = pd.concat([parent2item, asin2item], axis=0).drop_duplicates(subset=["asin_or_parent"])
    inter = inter.merge(map_df, on="asin_or_parent", how="left").dropna(subset=["item_id"])

    # 清理评分/时间
    if "rating" in inter: inter = inter[(inter["rating"]>=1) & (inter["rating"]<=5)]
    if pd.api.types.is_datetime64_any_dtype(inter["ts"]):
        inter["ts"] = inter["ts"].astype("int64") // 10**9
    elif inter["ts"].dtype == "O":
        inter["ts"] = pd.to_datetime(inter["ts"], errors="coerce")
        inter["ts"] = inter["ts"].astype("int64") // 10**9

    inter = inter[["user_id","item_id","rating","ts","rev_title","rev_text"]].reset_index(drop=True)

    # ---------- users ----------
    g = inter.groupby("user_id")["ts"]
    users = pd.DataFrame({
        "user_id": g.count().index,
        "hist_len": g.size().values,
        "first_ts": g.min().values,
        "last_ts": g.max().values
    })

    # 采样（可选）
    if args.sample_users > 0 and len(users) > args.sample_users:
        keep = users.sort_values("hist_len", ascending=False).head(args.sample_users)["user_id"]
        inter = inter[inter["user_id"].isin(keep)]
        users = users[users["user_id"].isin(keep)]

    os.makedirs(args.out, exist_ok=True)
    users.to_parquet(f"{args.out}/users.parquet", index=False, engine="pyarrow")
    items[["item_id","parent_asin","asin","title","description","category","price","image_urls"]].to_parquet(
        f"{args.out}/items.parquet", index=False, engine="pyarrow"
    )
    inter.to_parquet(f"{args.out}/interactions.parquet", index=False, engine="pyarrow")

    print("Saved to:", args.out)
    print("users:", len(users), "items:", len(items), "interactions:", len(inter))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", default="data/raw/Amazon_Fashion.jsonl")
    ap.add_argument("--meta",    default="data/raw/meta_Amazon_Fashion.jsonl")
    ap.add_argument("--out",     default="data/processed")
    ap.add_argument("--sample_users", type=int, default=0)
    args = ap.parse_args()
    main(args)
