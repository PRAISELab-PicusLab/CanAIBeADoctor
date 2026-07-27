"""
Merges the 5 per-model generated/<dataset>_<model>.csv files into the main
df_top200_<Dataset>.csv, one column per model/condition
(<model>_base/_empathy/_rephrase), keyed on `question`. Writes the merged
result back to df_top200_<Dataset>.csv (source columns are preserved; only
new columns are added).

Usage: python3 merge_generated_into_main.py medquad
       python3 merge_generated_into_main.py medredqa
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

MODELS = ["mixtral", "medgemma", "gptoss", "gemma4", "claude"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=list(MAIN_CSV))
    args = parser.parse_args()

    main_path = MAIN_CSV[args.dataset]
    main_df = pd.read_csv(main_path)
    main_df["question"] = main_df["question"].astype(str).str.strip()
    print(f"Main dataset: {main_df.shape}")

    for model in MODELS:
        gen_path = os.path.join(GEN_DIR, f"{args.dataset}_{model}.csv")
        if not os.path.exists(gen_path):
            print(f"  [skip] {gen_path} not found")
            continue
        gen_df = pd.read_csv(gen_path)
        gen_df["question"] = gen_df["question"].astype(str).str.strip()
        cond_cols = [c for c in gen_df.columns if c.startswith(f"{model}_")]
        gen_df = gen_df[["question"] + cond_cols]
        before = main_df.shape[1]
        main_df = main_df.merge(gen_df, on="question", how="left")
        print(f"  merged {model}: +{main_df.shape[1] - before} cols -> {list(cond_cols)}")
        for c in cond_cols:
            n_missing = main_df[c].isna().sum()
            if n_missing:
                print(f"    [warn] {c}: {n_missing} missing after merge")

    main_df.to_csv(main_path, index=False)
    print(f"\nSaved {main_df.shape} -> {main_path}")


if __name__ == "__main__":
    main()
