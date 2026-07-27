"""
Fills the small number of residual missing judge-score cells (rows where the
LLM judge produced malformed/unparseable output even after retry, e.g.
exhausted thinking-token budget) using the same similarity_pct -> 1-5 anchor
mapping the judge prompt itself uses for the factual axis (config_llm_judge's
similarity_bin): 0-20->1, 20-40->2, 40-60->3, 60-80->4, 80-100->5.

Applies to every "*_judge_{factual,completeness,coherence,safety}" column
still containing NaNs, using each model/condition's own "sim_medico_<col>"
similarity column as the source value. Writes filled values back into the
same *_with_judge.csv files, and also writes a filler note into the paired
"*_reason" column so the provenance is visible.

Usage: python3 fill_missing_judge_scores.py --dataset medquad_ext
       python3 fill_missing_judge_scores.py --dataset medredqa_ext
"""
import argparse
import os

import pandas as pd

from config_llm_judge import similarity_bin

HERE = os.path.dirname(__file__)

DATA_PATH = {
    "medquad_ext": os.path.join(HERE, "medquad_ext_with_judge.csv"),
    "medredqa_ext": os.path.join(HERE, "medredqa_ext_with_judge.csv"),
}

AXIS_SUFFIXES = ["judge_factual", "judge_completeness", "judge_coherence", "judge_safety"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATA_PATH))
    args = parser.parse_args()

    path = DATA_PATH[args.dataset]
    df = pd.read_csv(path)

    score_cols = [c for c in df.columns if any(c.endswith(s) for s in AXIS_SUFFIXES)]
    filled_total = 0
    for score_col in score_cols:
        n_missing = df[score_col].isna().sum()
        if n_missing == 0:
            continue
        # score_col looks like "<model>_<condition>_judge_<axis>"
        base_col = score_col.split("_judge_")[0]
        sim_col = f"sim_medico_{base_col}"
        if sim_col not in df.columns:
            print(f"  [warn] {score_col}: no similarity column {sim_col} found, skipping ({n_missing} missing)")
            continue
        reason_col = f"{score_col}_reason"
        mask = df[score_col].isna()
        sim_missing = df.loc[mask, sim_col].isna()
        if sim_missing.any():
            print(f"  [warn] {score_col}: {sim_missing.sum()} row(s) also missing {sim_col}, cannot fill those")
        fillable = mask & df[sim_col].notna()
        df.loc[fillable, score_col] = df.loc[fillable, sim_col].apply(similarity_bin)
        if reason_col in df.columns:
            df.loc[fillable, reason_col] = "filled from similarity_bin (judge output unparseable)"
        n_filled = fillable.sum()
        filled_total += n_filled
        print(f"  {score_col}: filled {n_filled}/{n_missing} from {sim_col}")

    df.to_csv(path, index=False)
    print(f"\nSaved {path} -- {filled_total} cells filled")

    remaining = df[score_cols].isna().sum()
    remaining = remaining[remaining > 0]
    if len(remaining):
        print("\n[warn] still missing after fill:")
        print(remaining)
    else:
        print("No missing values remain in any judge score column.")


if __name__ == "__main__":
    main()
