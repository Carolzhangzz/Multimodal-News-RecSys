# scripts/summarize_results.py
# -*- coding: utf-8 -*-
import os, glob
import pandas as pd
from pathlib import Path

RESULTS_DIR = "results"

def _load_csv(path, model_label=None):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["__source__"] = os.path.basename(path)
    if "HR@10" in df.columns:  # cold_eval
        df = df.rename(columns={"HR@10":"HR","NDCG@10":"NDCG"})
        df["K"] = df.get("K", 10)
    if "HR" not in df.columns or "NDCG" not in df.columns:
        return None
    if model_label:
        df["model"] = model_label
    return df

def _pick_content_from_fusion(fusion_df):
    # content-only == alpha_mode=const & alpha_const=0.0
    if fusion_df is None:
        return None
    m = (fusion_df["alpha_mode"]=="const") & (fusion_df["alpha_const"].fillna(1.0)==0.0)
    df = fusion_df[m].copy()
    if df.empty:
        return None
    df["model"] = "Content"
    return df

def _label_fusion_rows(df, prefix):
    if df is None:
        return None
    df = df.copy()
    # e.g., Fusion(linear,z,a=0.25) 或 Fusion(rrf,k=60)
    def name(row):
        if row.get("fusion","linear") == "rrf":
            return f"{prefix}(rrf,k={int(row.get('rrf_k',60))})"
        am = row.get("alpha_mode","const")
        if am == "const":
            return f"{prefix}(linear,{row.get('norm','none')},a={row.get('alpha_const',1.0)})"
        return f"{prefix}(linear,{row.get('norm','none')},dyn:t={row.get('alpha_t',3)},k={row.get('alpha_k',2)})"
    df["model"] = df.apply(name, axis=1)
    return df

def main():
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # 读取各日志（存在才并入）
    cold_eval = _load_csv("runs/mmgcn/cold_eval_log.csv", model_label="GCN")
    fusion_eval = _load_csv("runs/mmgcn/fusion_eval_log.csv")
    adapter_eval = _load_csv("runs/content_adapter/adapter_eval_log.csv")
    fusion_ad_eval = _load_csv("runs/mmgcn/fusion_adapter_eval_log.csv")

    # 从 fusion 里抽 content-only（a=0）
    content_only = _pick_content_from_fusion(fusion_eval)

    # 标注融合模型名
    fusion_eval_labeled = _label_fusion_rows(fusion_eval, "Fusion")
    fusion_ad_labeled   = _label_fusion_rows(fusion_ad_eval, "Fusion+Adapter")

    # 汇总
    frames = [x for x in [cold_eval, content_only, adapter_eval, fusion_eval_labeled, fusion_ad_labeled] if x is not None]
    if not frames:
        print("No logs found.")
        return
    all_df = pd.concat(frames, ignore_index=True, sort=False)

    # 只保留关键列
    keep_cols = ["time","model","split","n","HR","NDCG","K","n_neg",
                 "alpha_mode","alpha_const","fusion","norm","rrf_k",
                 "cold_user_th","cold_item_th","extreme_user","extreme_item","__source__"]
    for c in keep_cols:
        if c not in all_df.columns:
            all_df[c] = None
    all_df = all_df[keep_cols].sort_values(["split","model","time"])

    # 保存原始汇总
    out_csv = os.path.join(RESULTS_DIR, "summary_all.csv")
    all_df.to_csv(out_csv, index=False)

    # 生成每个 split 的 Markdown 表（HR/NDCG）
    for split, g in all_df.groupby("split"):
        piv_hr = g.pivot_table(index="model", values="HR", aggfunc="max")
        piv_nd = g.pivot_table(index="model", values="NDCG", aggfunc="max")
        # 对齐并合成
        table = piv_hr.join(piv_nd, how="outer").sort_values("HR", ascending=False)
        md_path = os.path.join(RESULTS_DIR, f"summary_{split}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {split}\n\n")
            f.write(table.to_markdown(floatfmt=".4f"))
            f.write("\n")
        print(f"Wrote {md_path}")

    # 生成一个“相对 GCN 提升”的表（按 split）
    rel_rows = []
    for split, g in all_df.groupby("split"):
        base = g[g["model"]=="GCN"][["HR","NDCG"]].max()
        if base.isna().any():
            continue
        hr0, nd0 = float(base["HR"]), float(base["NDCG"])
        g2 = g.groupby("model")[["HR","NDCG"]].max().reset_index()
        g2["HR_gain_%"]   = (g2["HR"]   - hr0) / max(hr0, 1e-8) * 100.0
        g2["NDCG_gain_%"] = (g2["NDCG"] - nd0) / max(nd0, 1e-8) * 100.0
        g2.insert(0, "split", split)
        rel_rows.append(g2)
    if rel_rows:
        gains = pd.concat(rel_rows, ignore_index=True)
        gains.to_csv(os.path.join(RESULTS_DIR, "summary_gains_vs_gcn.csv"), index=False)

    print(f"Saved combined CSV: {out_csv}")
    print("Done.")

if __name__ == "__main__":
    main()
