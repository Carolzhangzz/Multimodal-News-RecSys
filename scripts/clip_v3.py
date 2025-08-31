import argparse, os, io, re, time, ast, warnings
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import requests
import multiprocessing as mp

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# open_clip 用于 CLIP 文本/图像统一编码
import open_clip


# =====================
# 通用工具
# =====================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def to_plain_text(x):
    """把 list/dict/NaN 统一转成可读字符串，供 CLIP 文本编码使用。"""
    if isinstance(x, list):
        return " ".join(str(t) for t in x if t)
    if isinstance(x, dict):
        return " ".join(str(v) for v in x.values() if v)
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)

def sanitize_item_id(v: str) -> str:
    """文件名安全处理（Amazon ASIN/parent_asin 通常已安全，此处兜底）。"""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", str(v))

def _target_path(img_dir: str, item_id: str) -> str:
    return os.path.join(img_dir, f"{sanitize_item_id(item_id)}.jpg")


# =====================
# URL 标准化 & 主图挑选
# =====================

def _norm_url(u: str) -> str:
    u = u.strip()
    if u.startswith("http://"):
        u = "https://" + u[len("http://"):]
    return u if u.startswith(("http://","https://")) else ""

def _to_url_list(v):
    # v 可能是 list / tuple / np.ndarray / 字符串化的列表 / 单个URL / NaN
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        out = []
        for u in v:
            if isinstance(u, str) and u:
                nu = _norm_url(u)
                if nu: out.append(nu)
        return out
    if isinstance(v, str):
        # 尝试把 "['...','...']" 解析成列表
        try:
            vv = ast.literal_eval(v)
            return _to_url_list(vv)
        except Exception:
            nu = _norm_url(v)
            return [nu] if nu else []
    return []

def _pick_primary_from_list(urls: List[str]) -> Optional[str]:
    if not urls: return None
    # 优先高清/大图，其次避开 SRxx,yy 小缩略图
    hi_prefs = ("SL1600", "_UL1500_", "_UL1200_", "hi_res", "large")
    for p in hi_prefs:
        for u in urls:
            if p.lower() in u.lower():
                return u
    for u in urls:
        if "_SR" not in u.upper():   # 避开诸如 _SR38,50_ 的缩略图
            return u
    return urls[0]

def pick_primary_url(row):
    # 先用 image_url_primary，其次从 image_urls 里挑
    u = row.get("image_url_primary", None)
    if isinstance(u, str) and u.strip():
        return _norm_url(u)
    urls = _to_url_list(row.get("image_urls", []))
    return _pick_primary_from_list(urls)


# =====================
# （可选）下载函数仍保留，但默认关闭
# =====================

def _download_one(args: Tuple[str, str, str, float, int, bool]) -> Tuple[str, Optional[str], Optional[str]]:
    """(item_id, url, img_dir, timeout, retries, verify_ssl) -> (item_id, path_or_None, err)"""
    item_id, url, img_dir, timeout, retries, verify_ssl = args
    if not url:
        return item_id, None, "empty_url"

    fn = _target_path(img_dir, item_id)
    if os.path.exists(fn):
        return item_id, fn, None

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Connection": "close",
    }

    err = None
    for t in range(max(1, retries)):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, verify=verify_ssl)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img.save(fn, format="JPEG")
            return item_id, fn, None
        except Exception as e:
            err = str(e)
            time.sleep(0.1 * (t + 1))  # 轻微退避
    return item_id, None, err

def download_images_mp_stream(item_ids, urls, img_dir,
                              workers=4, timeout=10.0, retries=2, verify_ssl=True,
                              chunksize=64, fail_log_path=None):
    """流式喂任务到进程池。仅在未指定 --use-existing-only 时使用。"""
    ensure_dir(img_dir)
    total = len(item_ids)
    failures = []

    def _args_iter():
        for iid, url in zip(item_ids, urls):
            yield (iid, url, img_dir, timeout, retries, verify_ssl)

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max(1, workers), maxtasksperchild=200) as pool:
        for iid, path, err in tqdm(
            pool.imap_unordered(_download_one, _args_iter(), chunksize=chunksize),
            total=total, desc="Download imgs", mininterval=0.2
        ):
            if err:
                failures.append((iid, err))

    if fail_log_path and failures:
        pd.DataFrame(failures, columns=["item_id", "error"]).to_csv(fail_log_path, index=False)

    return failures


# =====================
# CLIP 编码（open_clip，文本/图像同一模型）
# =====================

class _ImagePathDataset(Dataset):
    def __init__(self, paths, preprocess):
        self.paths = paths
        self.preprocess = preprocess
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        if p is None or not os.path.exists(p):
            return torch.zeros(3, 224, 224)
        try:
            img = Image.open(p).convert("RGB")
            return self.preprocess(img)
        except Exception:
            return torch.zeros(3, 224, 224)

class CLIPEncoder:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        # 加速：TF32 + compile（可用即用）
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            self.model = torch.compile(self.model, mode="max-autotune")
        except Exception:
            pass
        # 混合精度（仅在 CUDA 上使用）
        self.use_fp16 = (self.device.type == "cuda")
        self.txt_dim = self.model.text_projection.shape[1] if hasattr(self.model, "text_projection") else self.model.text.head.out_features
        self.img_dim = getattr(self.model.visual, "output_dim", 512)

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        feats = []
        # tqdm 按样本数推进
        with tqdm(total=len(texts), desc="CLIP text", unit="txt", mininterval=0.1) as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                tokens = self.tokenizer(batch).to(self.device)
                if self.use_fp16:
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                        f = self.model.encode_text(tokens)
                else:
                    f = self.model.encode_text(tokens)
                f = F.normalize(f, dim=-1)
                feats.append(f.cpu().numpy().astype(np.float32))
                pbar.update(len(batch))
        return np.vstack(feats) if feats else np.zeros((0, self.txt_dim), dtype=np.float32)

    @torch.no_grad()
    def encode_images(self, paths: List[Optional[str]], batch_size: int = 128, workers: int = 4) -> np.ndarray:
        ds = _ImagePathDataset(paths, self.preprocess)
        dl = DataLoader(ds, batch_size=batch_size, num_workers=max(0, workers), pin_memory=True,
                        prefetch_factor=2 if workers > 0 else None, drop_last=False)
        feats = []
        done = 0
        with tqdm(total=len(paths), desc="CLIP image", unit="img", mininterval=0.1) as pbar:
            for batch in dl:
                batch = batch.to(self.device, non_blocking=True)
                if self.use_fp16:
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                        f = self.model.encode_image(batch)
                else:
                    f = self.model.encode_image(batch)
                f = F.normalize(f, dim=-1)
                feats.append(f.cpu().numpy().astype(np.float32))
                done += batch.size(0)
                pbar.update(batch.size(0))
        return np.vstack(feats) if feats else np.zeros((len(paths), self.img_dim), dtype=np.float32)


# =====================
# 主流程
# =====================

def build_texts_for_clip(items_df: pd.DataFrame) -> List[str]:
    title = items_df.get("title", pd.Series([""]*len(items_df))).apply(to_plain_text)
    desc  = items_df.get("description", pd.Series([""]*len(items_df))).apply(to_plain_text)
    cat   = items_df.get("category", pd.Series([""]*len(items_df))).apply(to_plain_text)
    texts = (title + " [TITLE] " + desc + " [CATEGORY] " + cat).str.slice(0, 512).tolist()
    return texts

def main(args):
    warnings.filterwarnings("ignore", category=UserWarning)  # 清理 PIL/mpl 等无关告警
    ensure_dir(args.out); ensure_dir(args.img_dir)

    items = pd.read_parquet(args.items, engine="pyarrow")
    inter = pd.read_parquet(args.interactions, engine="pyarrow")

    # 统一列存在且类型正确
    if "image_urls" not in items.columns:
        items["image_urls"] = [[] for _ in range(len(items))]
    else:
        items["image_urls"] = items["image_urls"].apply(_to_url_list)

    if "image_url_primary" not in items.columns:
        items["image_url_primary"] = None

    # 对缺失主图的行，从 image_urls 中挑一个主图
    need_primary = items["image_url_primary"].isna()
    items.loc[need_primary, "image_url_primary"] = items.loc[need_primary, "image_urls"].apply(_pick_primary_from_list)

    # 仅保留在 interactions 出现过的 item（强烈建议，能极大减小规模）
    if args.only_interacted:
        keep = set(inter["item_id"].unique())
        items = items[items["item_id"].isin(keep)].copy()

    # 限量（快速验证）
    if args.limit_items > 0:
        items = items.head(args.limit_items).copy()

    # 主图 URL & item_id
    item_ids = items["item_id"].astype(str).tolist()
    urls = items["image_url_primary"].apply(lambda u: _norm_url(u) if isinstance(u, str) else "").tolist()

    img_paths, has_image_flags = [], []

    if args.use_existing_only:
        # ✅ 仅用现有图片，不下载
        for iid in item_ids:
            p = _target_path(args.img_dir, iid)
            if os.path.exists(p):
                img_paths.append(p); has_image_flags.append(True)
            else:
                img_paths.append(None); has_image_flags.append(False)
        failures = []
    else:
        # （可选）仍支持下载模式
        failures = download_images_mp_stream(
            item_ids, urls, args.img_dir,
            workers=args.workers, timeout=args.timeout, retries=args.retries,
            verify_ssl=not args.no_ssl_verify,
            chunksize=64,
            fail_log_path=os.path.join(args.out, "download_failures.csv"),
        )
        for iid in item_ids:
            p = _target_path(args.img_dir, iid)
            if os.path.exists(p):
                img_paths.append(p); has_image_flags.append(True)
            else:
                img_paths.append(None); has_image_flags.append(False)

    # 可视化统计
    total = len(item_ids)
    have = sum(has_image_flags)
    print(f"[Stats] items: {total} | images ready: {have} ({have/total:.2%}) | will encode images only for existing files.")

    # =====================
    # 2) CLIP 文本/图像编码（带进度）
    # =====================
    encoder = CLIPEncoder(model_name=args.clip_model, pretrained=args.clip_pretrained, device=args.device)

    texts = build_texts_for_clip(items)
    txt_emb = encoder.encode_texts(texts, batch_size=args.txt_bs)

    if args.no_image:
        img_emb = np.zeros((len(items), encoder.img_dim), dtype=np.float32)
    else:
        # 仅对已存在图片跑 CLIP，其他行保持 0 向量
        valid_idx = [i for i, p in enumerate(img_paths) if p is not None]
        img_emb = np.zeros((len(items), encoder.img_dim), dtype=np.float32)
        if len(valid_idx) > 0:
            valid_paths = [img_paths[i] for i in valid_idx]
            valid_feats = encoder.encode_images(valid_paths, batch_size=args.img_bs, workers=args.workers)
            img_emb[np.array(valid_idx)] = valid_feats

    # =====================
    # 3) 结构化分支（示例：price -> log1p）
    # =====================
    price_series = items.get("price", pd.Series([0.0]*len(items))).astype(float)
    price = np.log1p(price_series.fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float32))

    # =====================
    # 4) 保存产物
    # =====================
    out_df = pd.DataFrame({
        "item_id": item_ids,
        f"clip_txt_{txt_emb.shape[1]}": [v.tolist() for v in txt_emb],
        f"clip_img_{img_emb.shape[1]}": [v.tolist() for v in img_emb],
        "price_log1p": price.tolist(),
        "has_image": has_image_flags,
        "image_path": img_paths,
    })
    out_df.to_parquet(f"{args.out}/item_features_clip.parquet", engine="pyarrow", index=False)

    # 用户统计特征（可用于冷启动/正则）
    g = inter.groupby("user_id")
    user_stats = pd.DataFrame({
        "user_id": g.size().index.astype(str),
        "hist_len": g.size().values,
        "mean_rating": g["rating"].mean().fillna(0).values,
        "last_ts": g["ts"].max().values,
    })
    user_stats.to_parquet(f"{args.out}/user_stats.parquet", engine="pyarrow", index=False)

    print("Saved:", f"{args.out}/item_features_clip.parquet", "and", f"{args.out}/user_stats.parquet")
    print("items encoded:", len(items), " | failures:", len(failures))


if __name__ == "__main__":
    mp.freeze_support()  # Windows/Colab 友好
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/processed/items.parquet")
    ap.add_argument("--interactions", default="data/processed/interactions.parquet")
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--img_dir", default="data/images")

    # 下载（默认禁用，除非取消 use-existing-only）
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--no-ssl-verify", dest="no_ssl_verify", action="store_true")
    ap.add_argument("--use-existing-only", action="store_true",
                    help="仅使用已下载的图片进行编码，不下载新图片；其余物品图像向量置零")

    # 采样
    ap.add_argument("--only-interacted", action="store_true")
    ap.add_argument("--limit-items", type=int, default=0)

    # CLIP
    ap.add_argument("--clip-model", default="ViT-B-32")
    ap.add_argument("--clip-pretrained", default="openai")
    ap.add_argument("--device", default=None, help="cuda|cpu，如未指定则自动选择")
    ap.add_argument("--txt-bs", type=int, default=512)
    ap.add_argument("--img-bs", type=int, default=128)
    ap.add_argument("--no-image", action="store_true")

    args = ap.parse_args()
    main(args)
