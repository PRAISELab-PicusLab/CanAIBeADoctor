"""
Builds the combined ablation-comparison dataset for one source dataset:
Main (v2 production prompt, already generated in the main run) vs Variant B
vs Variant C (config_prompts.ABLATION_VARIANTS), for Mixtral and MedGemma
only, all 3 conditions (base/empathy/rephrase) -- 2 models x 3 conditions x
3 prompt-versions = 18 text columns + reference_answer.

Reads: df_top200_<Dataset>.csv (for question, Answer_medico, and the 6 "Main"
columns mixtral_base/empathy/rephrase + medgemma_base/empathy/rephrase) and
generated/ablation_<dataset>_{mixtral,medgemma}.csv (for the 12
"_variantB"/"_variantC" columns).

Writes: df_ablation_<Dataset>.csv (question, Answer_medico, 18 text columns).
No metrics yet -- compute_readability.py/compute_similarity.py/
compute_sentiment.py/compute_emotions_ext.py run against config_ablation_*.py
add those afterward (recomputes metrics for the 6 Main columns too, even
though they're already computed in df_top200 -- redundant but cheap and
keeps this dataset fully self-contained).

Usage: python3 build_ablation_dataset.py --dataset medquad
       python3 build_ablation_dataset.py --dataset medredqa
"""
import argparse
import os

import pandas as pd

HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(HERE)
GEN_DIR = os.path.join(HERE, "generated")

MAIN_CSV = {
    "medquad": os.path.join(PROJECT_ROOT, "df_top200_MedQuAD.csv"),
    "medredqa": os.path.join(PROJECT_ROOT, "df_top200_MedRedQA.csv"),
}
OUT_CSV = {
    "medquad": os.path.join(PROJECT_ROOT, "df_ablation_MedQuAD.csv"),
    "medredqa": os.path.join(PROJECT_ROOT, "df_ablation_MedRedQA.csv"),
}

MODELS = ["mixtral", "medgemma"]
CONDITIONS = ["base", "empathy", "rephrase"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(MAIN_CSV))
    args = parser.parse_args()

    main_df = pd.read_csv(MAIN_CSV[args.dataset])
    main_df["question"] = main_df["question"].astype(str).str.strip()
    main_cols = ["question", "Answer_medico"] + [
        f"{m}_{c}" for m in MODELS for c in CONDITIONS
    ]
    missing = [c for c in main_cols if c not in main_df.columns]
    if missing:
        raise SystemExit(f"Missing expected main columns: {missing}")
    out = main_df[main_cols].copy()
    print(f"Main columns: {out.shape}")

    for model in MODELS:
        # Two possible sources per (condition, variant) column, newest wins:
        #  1. new-style per-shard file: ablation_<dataset>_<model>_<condition>_variant<X>.csv
        #     (single variant column; used since the 2026-07-24 race-condition fix)
        #  2. old-style shared file: ablation_<dataset>_<model>.csv (all 6 variant
        #     columns together; still relied on for combos verified clean before
        #     the fix -- e.g. medquad/mixtral, medquad/medgemma, and the 5 intact
        #     columns of medredqa/mixtral)
        shared_path = os.path.join(GEN_DIR, f"ablation_{args.dataset}_{model}.csv")
        shared_df = pd.read_csv(shared_path) if os.path.exists(shared_path) else None
        if shared_df is not None:
            shared_df["question"] = shared_df["question"].astype(str).str.strip()

        merged_cols = []
        for condition in CONDITIONS:
            for variant in ("B", "C"):
                col = f"{model}_{condition}_variant{variant}"
                shard_path = os.path.join(
                    GEN_DIR, f"ablation_{args.dataset}_{model}_{condition}_variant{variant}.csv"
                )
                if os.path.exists(shard_path):
                    shard_df = pd.read_csv(shard_path)
                    shard_df["question"] = shard_df["question"].astype(str).str.strip()
                    if col not in shard_df.columns:
                        print(f"  [warn] {shard_path} exists but missing column {col}")
                        continue
                    out = out.merge(shard_df[["question", col]], on="question", how="left")
                    merged_cols.append(f"{col} (per-shard file)")
                elif shared_df is not None and col in shared_df.columns:
                    out = out.merge(shared_df[["question", col]], on="question", how="left")
                    merged_cols.append(f"{col} (shared file)")
                else:
                    print(f"  [warn] {model}: no source found for {col} -- generation not finished yet")
        print(f"  merged {model}: +{len(merged_cols)} cols")
        for c in merged_cols:
            print(f"    {c}")

    out_path = OUT_CSV[args.dataset]
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out.shape} -> {out_path}")

    all_text_cols = main_cols[2:] + [c for c in out.columns if "variant" in c]
    n_missing = out[all_text_cols].isna().sum()
    if n_missing.sum():
        print("\n[warn] missing values per column (generation likely still in progress):")
        print(n_missing[n_missing > 0])


if __name__ == "__main__":
    main()
