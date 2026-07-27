"""
Builds an Excel sheet for human expert evaluation, mirroring the paper's
Appendix B.2 "Expert Evaluation Form" structure: question + candidate answer,
followed by three empty 1-5 Likert-scale columns (Clinical Accuracy,
Stylistic Appropriateness, Linguistic Precision) for a rater to fill in.

Samples N_QUESTIONS random questions from the given dataset and includes
every currently-active model/condition in MODEL_ORDER (Physicians reference
+ all non-excluded model/condition combos), one row per (question, model).

Usage: python3 build_expert_eval_sheet.py --dataset medquad_ext --n 2 --seed 42
"""
import argparse

import pandas as pd
from openpyxl.styles import Alignment


def load_config(dataset):
    if dataset == "medquad_ext":
        import config_medquad_ext as cfg
    elif dataset == "medredqa_ext":
        import config_medredqa_ext as cfg
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["medquad_ext", "medredqa_ext"])
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = load_config(args.dataset)

    df = pd.read_csv(cfg.DATA_PATH)
    sample = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)

    rows = []
    for qi, row in sample.iterrows():
        question = row["question"]
        for label in cfg.MODEL_ORDER:
            text_col = cfg.MODEL_COLUMNS[label]["text"]
            rows.append({
                "Question #": qi + 1,
                "Question": question,
                "Model": label.replace("\n", " "),
                "Answer": row[text_col],
                "Clinical Accuracy (1-5)": None,
                "Stylistic Appropriateness (1-5)": None,
                "Linguistic Precision (1-5)": None,
                "Notes": None,
            })
    out_df = pd.DataFrame(rows)

    out_path = args.out or f"expert_eval_{args.dataset}_n{args.n}.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name="Expert Evaluation")
        ws = writer.sheets["Expert Evaluation"]
        widths = {"A": 10, "B": 45, "C": 18, "D": 70, "E": 16, "F": 20, "G": 16, "H": 20}
        for col, w in widths.items():
            ws.column_dimensions[col].width = w
        for row_cells in ws.iter_rows(min_row=2, max_row=ws.max_row):
            for cell in row_cells:
                cell.alignment = Alignment(wrap_text=True, vertical="top")

    print(f"Saved {out_path} ({len(out_df)} rows = {args.n} questions x {len(cfg.MODEL_ORDER)} models)")
    print("Sampled questions:")
    for qi, row in sample.iterrows():
        print(f"  #{qi+1}: {row['question']}")


if __name__ == "__main__":
    main()
