"""
One-off backfill: adds a "reference_answer" column (the physician's original
answer, Answer_medico) to already-existing generated/<dataset>_<model>.csv
files that predate the fix in generate_answers.py/generate_claude.py (which
now adds this column automatically on every future write).

Only safe to run against a file once its writer process has finished --
running it against a file still being actively checkpointed by a live
generation job risks a lost-update race (read here, write there, whichever
finishes last wins and the other's update is silently dropped).

Usage: python3 backfill_reference_answer.py <dataset>_<model>.csv [more files...]
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(__file__)
GEN_DIR = os.path.join(HERE, "generated")

DATASETS = {
    "medquad": {
        "csv": os.path.join(os.path.dirname(HERE), "df_top200_MedQuAD.csv"),
        "question_col": "question", "answer_col": "Answer_medico",
    },
    "medredqa": {
        "csv": os.path.join(os.path.dirname(HERE), "df_top200_MedRedQA.csv"),
        "question_col": "question", "answer_col": "Answer_medico",
    },
}


def main():
    files = sys.argv[1:]
    if not files:
        print("Usage: python3 backfill_reference_answer.py <file1.csv> [file2.csv ...]")
        return
    for fname in files:
        path = os.path.join(GEN_DIR, fname) if not os.path.isabs(fname) else fname
        dataset = os.path.basename(path).split("_")[0]
        if dataset not in DATASETS:
            print(f"  [skip] {fname}: unknown dataset prefix '{dataset}'")
            continue
        ds_cfg = DATASETS[dataset]

        gen_df = pd.read_csv(path)
        if "reference_answer" in gen_df.columns and gen_df["reference_answer"].notna().all():
            print(f"  [skip] {fname}: already has reference_answer, fully populated")
            continue
        gen_df[ds_cfg["question_col"]] = gen_df[ds_cfg["question_col"]].astype(str).str.strip()
        if "reference_answer" in gen_df.columns:
            gen_df = gen_df.drop(columns=["reference_answer"])

        src_df = pd.read_csv(ds_cfg["csv"])
        src_df[ds_cfg["question_col"]] = src_df[ds_cfg["question_col"]].astype(str).str.strip()
        src_df = src_df.drop_duplicates(subset=[ds_cfg["question_col"]])
        ref_df = src_df[[ds_cfg["question_col"], ds_cfg["answer_col"]]].rename(
            columns={ds_cfg["answer_col"]: "reference_answer"}
        )

        merged = gen_df.merge(ref_df, on=ds_cfg["question_col"], how="left")
        merged.to_csv(path, index=False)
        n_missing = merged["reference_answer"].isna().sum()
        print(f"  [done] {fname}: {len(merged)} rows, {n_missing} missing reference_answer")


if __name__ == "__main__":
    main()
