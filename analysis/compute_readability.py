"""
Computes Flesch-Kincaid grade level + Gunning Fog index for every text column
in READABILITY_TARGETS (all 16 model/condition + reference columns). Pure
textstat, CPU-only, safe on the login node.

Usage: python3 compute_readability.py --dataset medquad_ext
       python3 compute_readability.py --dataset medredqa_ext
"""
import argparse

import pandas as pd
import textstat


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


def safe_fk(text):
    return textstat.flesch_kincaid_grade(str(text)) if isinstance(text, str) and text.strip() else None


def safe_gf(text):
    return textstat.gunning_fog(str(text)) if isinstance(text, str) and text.strip() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["medquad_ext", "medredqa_ext", "ablation_medquad", "ablation_medredqa"])
    args = parser.parse_args()
    cfg = load_config(args.dataset)

    df = pd.read_csv(cfg.DATA_PATH)
    print(f"Loaded {cfg.DATA_PATH}: {df.shape}")

    seen = set()
    for text_col, fk_col, gf_col in cfg.READABILITY_TARGETS:
        if text_col in seen or text_col not in df.columns:
            continue
        seen.add(text_col)
        df[fk_col] = df[text_col].apply(safe_fk)
        df[gf_col] = df[text_col].apply(safe_gf)
        print(f"  {text_col} -> {fk_col}, {gf_col} ({df[fk_col].notna().sum()} rows)")

    df.to_csv(cfg.DATA_PATH, index=False)
    print(f"Saved {cfg.DATA_PATH}")


if __name__ == "__main__":
    main()
