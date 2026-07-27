"""
Merges metrics/<dataset>_{similarity,sentiment,emotion}.csv (produced by
compute_similarity.py / compute_sentiment.py / compute_emotions_ext.py, safe
to run in parallel since each writes its own file) into the main dataset CSV
(cfg.DATA_PATH), keyed on `question`. Run this only after all three metric
jobs for a dataset have finished.

Usage: python3 merge_metrics.py --dataset medquad_ext
       python3 merge_metrics.py --dataset medredqa_ext
"""
import argparse
import os

import pandas as pd


def load_config(dataset):
    if dataset == "medquad_ext":
        import config_medquad_ext as cfg
    elif dataset == "medredqa_ext":
        import config_medredqa_ext as cfg
    elif dataset == "ablation_medquad":
        import config_ablation_medquad as cfg
    elif dataset == "ablation_medredqa":
        import config_ablation_medredqa as cfg
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=[
        "medquad_ext", "medredqa_ext", "ablation_medquad", "ablation_medredqa",
    ])
    args = parser.parse_args()
    cfg = load_config(args.dataset)

    df = pd.read_csv(cfg.DATA_PATH)
    df["question"] = df["question"].astype(str).str.strip()
    print(f"Main dataset: {df.shape}")

    metrics_dir = os.path.join(os.path.dirname(__file__), "metrics")
    for metric in ["similarity", "sentiment", "emotion"]:
        path = os.path.join(metrics_dir, f"{args.dataset}_{metric}.csv")
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        metric_df = pd.read_csv(path)
        metric_df["question"] = metric_df["question"].astype(str).str.strip()
        new_cols = [c for c in metric_df.columns if c != "question"]
        before = df.shape[1]
        df = df.merge(metric_df, on="question", how="left")
        print(f"  merged {metric}: +{df.shape[1] - before} cols")
        for c in new_cols:
            n_missing = df[c].isna().sum()
            if n_missing:
                print(f"    [warn] {c}: {n_missing} missing after merge")

    df.to_csv(cfg.DATA_PATH, index=False)
    print(f"\nSaved {df.shape} -> {cfg.DATA_PATH}")


if __name__ == "__main__":
    main()
