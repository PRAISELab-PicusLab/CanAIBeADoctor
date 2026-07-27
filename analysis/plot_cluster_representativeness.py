"""
Visualizes the readability-driven clustering selection (build_extended_medquad.py
/ build_extended_medredqa.py) for the two extended datasets: scatter of
Flesch-Kincaid vs Gunning-Fog for the FULL candidate source corpus (blue,
"All physician answers"), with the actual rows selected for the analyses
(df_top200_MedQuAD.csv / df_top200_MedRedQA.csv -- 231 / 200 rows
respectively) highlighted in orange ("Cluster representatives").

Does not re-run clustering -- the orange points are the real, already-used
selection, plotted against the same full-corpus background the original
selection scripts drew from, to visualize how well the selection covers the
full readability space.

Usage: python3 plot_cluster_representativeness.py
"""
import os

import numpy as np
import pandas as pd
import textstat
import matplotlib.pyplot as plt

import common

HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(HERE)

MEDQUAD_HF_CSV_URL = "https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset/resolve/main/medDataset_processed.csv"
MEDREDQA_DATA_PATH = os.path.join(PROJECT_ROOT, "medredqa", "medredqa_train.csv")

MEDQUAD_SELECTED_PATH = os.path.join(PROJECT_ROOT, "df_top200_MedQuAD.csv")
MEDREDQA_SELECTED_PATH = os.path.join(PROJECT_ROOT, "df_top200_MedRedQA.csv")

MEDQUAD_OUT = os.path.join(HERE, "outputs", "medquad_ext", "scatter_cluster_representativeness.pdf")
MEDREDQA_OUT = os.path.join(HERE, "outputs", "medredqa_ext", "scatter_cluster_representativeness.pdf")

MIN_WORDS = 50
MAX_QMARKS = 2


def iqr_filter(df, cols):
    mask = pd.Series(True, index=df.index)
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask &= df[c].between(lo, hi)
    return df[mask]


def plot_representativeness(pool_fk, pool_gf, sel_fk, sel_gf, title, out_path, figsize=(14, 10), dpi=300):
    """common.severity_cluster_scatter's (8,6) figsize is small compared to
    the rest of the outputs/ battery (bar/heatmap plots run 20-24in wide) --
    scaled up here so this figure isn't the odd one out on the page."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.scatter(pool_fk, pool_gf, s=45, alpha=0.15, color="#4C72B0", label="All physician answers")
    plt.scatter(sel_fk, sel_gf, s=200, alpha=0.9, color="#DD8452", edgecolor="black",
                linewidth=1.5, zorder=3, label=f"Cluster representatives ({len(sel_fk)})")
    plt.title(title, fontweight="bold", fontsize=20)
    plt.xlabel("Flesch-Kincaid", fontsize=18)
    plt.ylabel("Gunning-Fog", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(frameon=True, framealpha=1, edgecolor="black", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved {out_path}")


def do_medquad():
    print("Loading MedQuAD full corpus ...")
    full = pd.read_csv(MEDQUAD_HF_CSV_URL)
    full = full.rename(columns={"Question": "question", "Answer": "Answer_medico"})
    full["question"] = full["question"].astype(str).str.strip()
    full = full.dropna(subset=["question", "Answer_medico"])
    full = full.drop_duplicates(subset=["question"])
    print("Full corpus:", full.shape)

    print("Computing FKGL/GFI ...")
    full["_fk"] = full["Answer_medico"].apply(lambda t: textstat.flesch_kincaid_grade(str(t)))
    full["_gf"] = full["Answer_medico"].apply(lambda t: textstat.gunning_fog(str(t)))
    filtered = iqr_filter(full, ["_fk", "_gf"])
    print("After IQR filtering:", filtered.shape)

    selected = pd.read_csv(MEDQUAD_SELECTED_PATH)
    print("Selected (used in analyses):", selected.shape)

    # "How to prevent Parasites - Cysticercosis ?" (FK=49.43, GF=51.95) is a
    # textstat outlier far outside the rest of the selection (next-highest GF
    # is ~23.5) -- excluded from this plot only (display artifact), not from
    # the underlying dataset used in the actual analyses.
    outlier_mask = selected["question"].str.contains("How to prevent Parasites - Cysticercosis", na=False)
    print(f"Excluding {outlier_mask.sum()} outlier row(s) from the plot: "
          f"{selected.loc[outlier_mask, 'question'].tolist()}")
    selected_plot = selected[~outlier_mask]

    plot_representativeness(
        filtered["_fk"], filtered["_gf"],
        selected_plot["Answer_medico_FleschKincaid"], selected_plot["Answer_medico_GunningFog"],
        f"MedQuAD Extended: Clustering-based representative selection ({len(selected_plot)} reps)",
        MEDQUAD_OUT,
    )


def do_medredqa():
    print("Loading MedRedQA full corpus ...")
    full = pd.read_csv(MEDREDQA_DATA_PATH)
    full = full.rename(columns={"Title": "question", "Body": "question_text", "Response": "Answer_medico"})
    full["question"] = full["question"].astype(str).str.strip()
    full = full.dropna(subset=["question", "Answer_medico"])
    full = full.drop_duplicates(subset=["question", "Answer_medico"])
    print("Full corpus:", full.shape)

    resp = full["Answer_medico"].astype(str)
    word_count = resp.str.split().str.len()
    ends_with_q = resp.str.strip().str.endswith("?")
    qmark_count = resp.str.count(r"\?")
    keep = (word_count >= MIN_WORDS) & (~ends_with_q) & (qmark_count < MAX_QMARKS)
    full = full[keep].copy()
    print(f"After long-genuine-answer filter: {full.shape}")

    print("Computing FKGL/GFI ...")
    full["_fk"] = full["Answer_medico"].apply(lambda t: textstat.flesch_kincaid_grade(str(t)))
    full["_gf"] = full["Answer_medico"].apply(lambda t: textstat.gunning_fog(str(t)))
    filtered = iqr_filter(full, ["_fk", "_gf"])
    print("After IQR filtering:", filtered.shape)

    selected = pd.read_csv(MEDREDQA_SELECTED_PATH)
    print("Selected (used in analyses):", selected.shape)

    plot_representativeness(
        filtered["_fk"], filtered["_gf"],
        selected["Answer_medico_FleschKincaid"], selected["Answer_medico_GunningFog"],
        f"MedRedQA Extended: Clustering-based representative selection ({len(selected)} reps)",
        MEDREDQA_OUT,
    )


def main():
    common.apply_style()
    do_medquad()
    do_medredqa()


if __name__ == "__main__":
    main()
